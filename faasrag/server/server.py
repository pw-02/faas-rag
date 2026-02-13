import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pathlib import Path
import grpc
import hydra
from omegaconf import OmegaConf
import torch
from concurrent.futures import ThreadPoolExecutor
from faasrag.core.rag_pipeline import RagPipeline
from faasrag.core.args import RagServiceConfig
import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc
# Background resource sampler (writes resource_usage.jsonl every N seconds)
from faasrag.core.resource_usage import resource_monitor_loop

    
def save_cfg(cfg, path: str = "resolved.yaml") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

def setup_logger(
    name: str,
    log_to_file: bool = False,
    log_file: str = "rag_service.log",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        ch = logging.StreamHandler()  # stderr
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_to_file and log_file:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:
        # update levels if already initialized
        for h in logger.handlers:
            h.setLevel(level)

    return logger



def _parse_log_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    return logging.INFO


@dataclass
class Job:
    request: rag_pb2.RAGRequest
    future: asyncio.Future
    arrival_ts: float

def resolve_device(device):
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:0" if device == "auto" else device


class ScheduledRAGService(rag_pb2_grpc.RAGServiceServicer):
    """
    Scheduler-shaped RAG service:
    - Query() enqueues a job and awaits completion
    - background worker(s) pop jobs and run RagPipeline in a thread
    """
    def __init__(
        self,
        cfg: RagServiceConfig,
        *,
        num_workers: int,
        max_inflight: int,
        logger: Optional[logging.Logger] = None,
    ):
   
        cfg.embedder.device = resolve_device(cfg.embedder.device)
        cfg.generator.device = resolve_device(cfg.generator.device)
        self.args = cfg
        if cfg.embedder.device == "cpu" or cfg.generator.device == "cpu":
            logger.warning("CUDA not available, running on CPU (this may be slow)")

        logger.info("Using embedder device: %s", cfg.embedder.device)
        logger.info("Using generator device: %s", cfg.generator.device)

        self.index_vector_count = getattr(cfg.index, "index_vector_count", None)

        if cfg.telemetry.enabled:
            # logger.info("Telemetry enabled: writing resource usage logs to %s", cfg.telemetry.path)
            save_cfg(cfg, path=os.path.join(cfg.telemetry.dir, "service_config.yaml"))

        self.logger = logger
        self.rag_pipeline = RagPipeline(
            generator_cfg=cfg.generator,
            embedder_cfg=cfg.embedder,
            index_cfg=cfg.index,
            cache_cfg=cfg.cache if hasattr(cfg, "cache") else None,
            docstore_cfg=cfg.docstore,
            docstore_backend=cfg.docstore_backend,
            artifact_dir=cfg.artifact_dir,
            top_k=cfg.top_k,
            retrieve_only=cfg.retrieve_only,
            build_prompt_method=cfg.prompt_build_method,
            max_ctx_chars=cfg.max_ctx_chars,
            seed=cfg.seed,
            logger=self.logger,
        )

        self._num_workers = max(1, int(num_workers))
        self._max_inflight = max(1, int(max_inflight))

        # Backpressure: cap queue size (optional but recommended)
        self.pending: asyncio.Queue[Job] = asyncio.Queue(maxsize=self._max_inflight * 4)

        # Limit concurrent *pipeline executions*
        self._inflight_sem = asyncio.Semaphore(self._max_inflight)

        # Thread pool: size to workers (avoid double-queueing inside executor)
        self._executor = ThreadPoolExecutor(max_workers=self._num_workers)

        self._worker_tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()

        self.logger.info(
            "Initialized RagPipeline top_k=%d workers=%d max_inflight=%d",
            cfg.top_k,
            self._num_workers,
            self._max_inflight,
        )

    def start(self) -> None:
        for i in range(self._num_workers):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._worker_tasks:
            t.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._executor.shutdown(wait=False)

    # ---- IMPORTANT: adapt this to your RagPipeline API ----
    def _run_pipeline_sync(self, query: str) -> Dict[str, Any]:
        """
        Run pipeline synchronously in a worker thread.
        """
        result = self.rag_pipeline.run(query=query)

        if isinstance(result, dict):
            return result
        # Minimal fallback
        return {"answer": str(result)}
    # ------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()

        while not self._stop.is_set():
            job: Job = await self.pending.get()
            
            # How long the request waited in the service queue before a worker started it
            queue_s = time.perf_counter() - job.arrival_ts

            # End-to-end time measured from when worker began handling it (not from client)
            t0 = time.perf_counter()

            try:
                req = job.request
                async with self._inflight_sem:
                    result = await loop.run_in_executor(
                    self._executor, self._run_pipeline_sync, 
                    req.query, 
                )

                t1 = time.perf_counter()
                e2e_s = t1 - t0
                timings_s = result.get("timings_s") or {}

                answer = str(result.get("answer", ""))
                
                retrieved_doc_ids = result.get("retrieved_doc_ids") or []
                prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
                completion_tokens = int(result.get("completion_tokens", 0) or 0)
                total_tokens = int(result.get("total_tokens", 0) or 0)

                # Pipeline stage timings are in seconds
                raw_timings = result.get("timings_s") or {}
                timings_s: dict[str, float] = {k: float(v) for k, v in raw_timings.items()}

                # Add service-level timings for trace visibility
                timings_s["queue_s"] = float(queue_s)
                timings_s["e2e_s"] = float(e2e_s)

                cache_hits = int(result.get("cache_hits", 0))
                cache_misses = int(result.get("cache_misses", 0))
                cache_used = bool(result.get("cache_used", False))

                # Optional: log e2e + queue for debugging
                self.logger.debug(
                "req done worker=%d queue_s=%.3f e2e_s=%.3f ann_s=%.3f docstore_s=%.3f decode_s=%.3f",
                worker_id,
                timings_s.get("queue_s", 0.0),
                timings_s.get("e2e_s", 0.0),
                timings_s.get("ann_s", 0.0),
                timings_s.get("docstore_s", 0.0),
                timings_s.get("decode_s", 0.0),
                )

                trace = rag_pb2.Trace(
                    timings_s=timings_s,
                    cache_hits=cache_hits,
                    cache_misses=cache_misses,
                    cache_used=cache_used,
                    k=int(self.rag_pipeline.top_k),
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(total_tokens),
                    retrieved_doc_ids=retrieved_doc_ids,
                    index_vector_count=int(self.index_vector_count) if self.index_vector_count is not None else 0,
                )
                if not job.future.done():
                    job.future.set_result(rag_pb2.RAGResponse(answer=answer, trace=trace))

            except Exception as e:
                self.logger.exception("Worker %d failed", worker_id)
                if not job.future.done():
                      job.future.set_exception(e)

            finally:
                self.pending.task_done()

    async def Query(self, request: rag_pb2.RAGRequest, context) -> rag_pb2.RAGResponse:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        job = Job(request=request, future=fut, arrival_ts=time.perf_counter())

        try:
            self.pending.put_nowait(job)
        except asyncio.QueueFull:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Server overloaded (queue full)")
            return rag_pb2.RAGResponse(answer="", trace=rag_pb2.Trace())

        try:
            remaining = context.time_remaining()
            if remaining is not None and remaining > 0:
                return await asyncio.wait_for(fut, timeout=remaining)
            return await fut

        except asyncio.TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Request deadline exceeded in scheduler")
            return rag_pb2.RAGResponse(answer="", trace=rag_pb2.Trace())

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return rag_pb2.RAGResponse(answer="", trace=rag_pb2.Trace())
    
    async def Ping(self, request: rag_pb2.PingRequest, context) -> rag_pb2.PingResponse:
        return rag_pb2.PingResponse(
            ok=True,
            queue_depth=self.pending.qsize(),
            max_inflight=self._max_inflight,
        )


async def _serve_async(cfg: RagServiceConfig):
    
    logger = setup_logger(
        name="rag_service",
        level=_parse_log_level(cfg.log_level))
    logger.info("Starting RAG Service")

    # Optional background resource monitor (telemetry)
    monitor_task: Optional[asyncio.Task] = None
    if cfg.telemetry is not None and cfg.telemetry.enabled:
        interval_s = float(cfg.telemetry.interval_s)
        out_path = os.path.join(cfg.telemetry.dir, "resource_usage.jsonl")

        monitor_task = asyncio.create_task(
            resource_monitor_loop(interval_s=interval_s, output_path=out_path)
        )
        logger.info("Telemetry enabled interval=%.2fs path=%s", interval_s, out_path)

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )
    service = ScheduledRAGService(
        cfg,
        num_workers=cfg.num_workers,
        max_inflight=cfg.max_inflight,
        logger=logger,
    )
    service.start()

    rag_pb2_grpc.add_RAGServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    await server.start()
    logger.info("gRPC server listening on %s:%s", cfg.host, cfg.port)

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.warning("⛔ Shutting down RAG Service...")
    finally:
        if monitor_task is not None:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)

        await service.stop()
        await server.stop(grace=None)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    asyncio.run(_serve_async(cfg))


if __name__ == "__main__":
    main()
