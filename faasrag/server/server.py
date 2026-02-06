import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import grpc
import hydra
import torch
from concurrent.futures import ThreadPoolExecutor

from faasrag.core.rag_pipeline import RagPipeline
from faasrag.core.args import RagServiceConfig
import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc
# Background resource sampler (writes resource_usage.jsonl every N seconds)
from faasrag.core.resource_usage import resource_monitor_loop


def setup_logger(
    name: str,
    log_file: str = "ragservice.log",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.hasHandlers():
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

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
        self.cfg = cfg
        device = cfg.device or "auto"
        self.cfg.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        self.logger = logger or logging.getLogger("rag_service")

        self.rag_pipeline = RagPipeline(
            generator_cfg=cfg.generator,
            embedder_cfg=cfg.embedder,
            index_cfg=cfg.index,
            cache_cfg=cfg.cache,
            docstore_cfg=cfg.docstore,
            artifact_dir=cfg.artifact_dir,
            top_k=cfg.top_k,
            device=self.cfg.device,
            retrieve_only=cfg.retrieve_only,
            prompt_type=cfg.prompt_type,
            max_ctx_chars=cfg.max_ctx_chars,
            seed=cfg.seed,
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
            "Initialized RagPipeline device=%s top_k=%d workers=%d max_inflight=%d",
            cfg.device,
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
    def _run_pipeline_sync(self, query: str, top_k: int, max_tokens: int) -> Dict[str, Any]:
        """
        Run pipeline synchronously in a worker thread.
        """
        result = self.rag_pipeline.run(query=query, top_k=top_k, max_tokens=max_tokens)

        if isinstance(result, dict):
            return result
        # Minimal fallback
        return {"answer": str(result)}
    # ------------------------------------------------------

    async def _worker_loop(self, worker_id: int) -> None:
        loop = asyncio.get_running_loop()

        while not self._stop.is_set():
            job: Job = await self.pending.get()
            t0 = time.perf_counter()

            try:
                req = job.request
                top_k = int(req.k) if req.k and req.k > 0 else int(self.cfg.top_k)
                max_tokens = int(req.max_tokens) if req.max_tokens and req.max_tokens > 0 else 0
                
                async with self._inflight_sem:
                    result = await loop.run_in_executor(
                        self._executor, self._run_pipeline_sync, req.query, top_k, max_tokens
                    )
                t1 = time.perf_counter()

                answer = str(result.get("answer", ""))
                retrieved_doc_ids = result.get("retrieved_doc_ids") or []
                prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
                output_tokens = int(result.get("output_tokens", 0) or 0)

                # Your pipeline returns timings_s (seconds). Convert to ms for protobuf Trace.
                timings_s = result.get("timings_s") or {}
                retrieve_ms = float(timings_s.get("total_retrieval_s", 0.0)) * 1000.0
                decode_ms = float(timings_s.get("decode_s", 0.0)) * 1000.0

                # End-to-end latency for safety if decode_ms missing
                e2e_ms = (t1 - t0) * 1000.0
                if decode_ms <= 0.0:
                    decode_ms = e2e_ms

                trace = rag_pb2.Trace(
                    retrieve_ms=retrieve_ms,
                    rerank_ms=float(result.get("rerank_ms", 0.0) or 0.0),
                    llm_queue_ms=float(result.get("llm_queue_ms", 0.0) or 0.0),
                    decode_ms=decode_ms,
                    k=top_k,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    retrieved_doc_ids=retrieved_doc_ids,
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

async def _serve_async(cfg: RagServiceConfig):
    
    logger = setup_logger(name="rag_service", level=_parse_log_level(cfg.log_level))
    logger.info("Starting RAG Service")

    # Optional background resource monitor (telemetry)
    monitor_task: Optional[asyncio.Task] = None
    if cfg.telemetry is not None and cfg.telemetry.enabled:
        interval_s = float(cfg.telemetry.interval_s)
        out_path = str(cfg.telemetry.path)

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
