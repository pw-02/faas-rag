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


def setup_logger(
    name: str,
    log_file: str = "ragservice.log",
    level=logging.INFO,
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


@dataclass
class Job:
    request: rag_pb2.RAGRequest
    future: asyncio.Future
    arrival_ts: float


class ScheduledRAGService(rag_pb2_grpc.RAGServiceServicer):
    """
    "Scheduler-shaped" RAG service:
    - Query() enqueues a job and awaits completion
    - background worker(s) pop jobs and run RagPipeline in a thread (non-blocking to event loop)
    """

    def __init__(
        self,
        cfg: RagServiceConfig,
        num_workers:int,
        max_inflight: int,        
        logger: Optional[logging.Logger] = None,

    ):
        self.cfg = cfg
        cfg.device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger("rag_service")

        self.rag_pipeline = RagPipeline(
            generator_cfg=cfg.generator,
            embedder_cfg=cfg.embedder,
            index_cfg=cfg.index,
            docstore_cfg=cfg.docstore,
            artifact_dir=cfg.artifact_dir,
            top_k=cfg.top_k,
            device=cfg.device,
        )

        self.logger.info(
            f"Initialized RagPipeline with device={cfg.device}, top_k={cfg.top_k}, num_workers={num_workers}, max_inflight={max_inflight}"
        )

        # Pending jobs queue
        self.pending: asyncio.Queue[Job] = asyncio.Queue()

        # Limit concurrent pipeline executions (important!)
        self._executor = ThreadPoolExecutor(max_workers=max_inflight)
        self._num_workers = max(1, int(num_workers))
        self._worker_tasks: List[asyncio.Task] = []
        self._stop = asyncio.Event()

    def start(self) -> None:
        # Must be called after the event loop exists (inside _serve_async)
        for i in range(self._num_workers):
            self._worker_tasks.append(asyncio.create_task(self._worker_loop(i)))

    async def stop(self) -> None:
        self._stop.set()
        for t in self._worker_tasks:
            t.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._executor.shutdown(wait=False)

    def _run_pipeline_sync(self, query: str, top_k: int) -> Dict[str, Any]:
        """
        Run pipeline synchronously. Adjust the call to match your RagPipeline API.
        """
        # ---- ADJUST THIS LINE if needed ----
        result = self.rag_pipeline(query, top_k=top_k)
        # -----------------------------------

        # Normalize to a dict so the rest of the code is stable
        if isinstance(result, str):
            return {"answer": result}
        if isinstance(result, dict):
            return result

        answer = getattr(result, "answer", None) or str(result)
        retrieved_doc_ids = getattr(result, "retrieved_doc_ids", None)
        return {"answer": answer, "retrieved_doc_ids": retrieved_doc_ids}

    async def _worker_loop(self, worker_id: int) -> None:
        while not self._stop.is_set():
            job: Job = await self.pending.get()
            t0 = time.perf_counter()

            try:
                req = job.request
                top_k = int(req.k) if req.k and req.k > 0 else int(self.cfg.top_k)
                max_tokens = int(req.max_tokens) if req.max_tokens and req.max_tokens > 0 else 0

                # Run blocking pipeline in bounded thread pool
                result = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._run_pipeline_sync, req.query, top_k
                )
                t1 = time.perf_counter()

                answer = result.get("answer", "")
                retrieved_doc_ids = result.get("retrieved_doc_ids") or []
                prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
                output_tokens = int(result.get("output_tokens", 0) or 0)

                e2e_ms = (t1 - t0) * 1000.0
                # Until RagPipeline exposes stage timings, we put all time in decode_ms
                trace = rag_pb2.Trace(
                    retrieve_ms=0.0,
                    rerank_ms=0.0,
                    llm_queue_ms=0.0,
                    decode_ms=e2e_ms,
                    k=top_k,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens if output_tokens > 0 else max_tokens,
                    retrieved_doc_ids=retrieved_doc_ids,
                )

                if not job.future.done():
                    job.future.set_result(rag_pb2.RAGResponse(answer=answer, trace=trace))

            except Exception as e:
                self.logger.exception(f"Worker {worker_id} failed")
                if not job.future.done():
                    job.future.set_exception(e)

            finally:
                self.pending.task_done()

    async def Query(self, request: rag_pb2.RAGRequest, context) -> rag_pb2.RAGResponse:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        job = Job(request=request, future=fut, arrival_ts=time.perf_counter())

        await self.pending.put(job)

        try:
            # Respect gRPC deadline if present
            remaining = context.time_remaining()
            if remaining is not None and remaining > 0:
                return await asyncio.wait_for(fut, timeout=remaining)
            return await fut #“I don’t have the answer yet. Please pause this request until this Future is filled.”
        except asyncio.TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Request deadline exceeded in scheduler")
            return rag_pb2.RAGResponse(answer="", trace=rag_pb2.Trace())
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return rag_pb2.RAGResponse(answer="", trace=rag_pb2.Trace())


async def _serve_async(cfg: RagServiceConfig):
    logger = setup_logger(name="rag_service", level=cfg.log_level)
    logger.info("Starting RAG Service")

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ])

    service = ScheduledRAGService(cfg, num_workers=cfg.num_workers, max_inflight=cfg.max_inflight, logger=logger)
    service.start()

    rag_pb2_grpc.add_RAGServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    await server.start()
    logger.info(f"gRPC server listening on {cfg.host}:{cfg.port}")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.warning("⛔ Shutting down RAG Service...")
    finally:
        await service.stop()
        await server.stop(grace=None)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    asyncio.run(_serve_async(cfg))


if __name__ == "__main__":
    main()
