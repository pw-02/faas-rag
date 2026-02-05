import asyncio
import random
import time
from typing import List

import grpc

import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc

def ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

class RAGService(rag_pb2_grpc.RAGServiceServicer):
    async def Query(self, request: rag_pb2.RAGRequest, context) -> rag_pb2.RAGResponse:
        # Simulate a pipeline: retrieve -> rerank -> queue -> decode
        # Make times depend on k and query_class to create variance & tail latency.
        k = request.k or 20
        qc = request.query_class or "medium"

        # synthetic distributions
        base_retrieve = {"short": 8, "medium": 20, "long": 40}.get(qc, 20)
        base_decode = {"short": 120, "medium": 300, "long": 600}.get(qc, 300)

        # retrieval scales with k
        retrieve_ms = max(1.0, random.gauss(base_retrieve + 0.7 * k, 4.0))
        rerank_ms = max(0.0, random.gauss(0.2 * k, 3.0))  # optional stage
        llm_queue_ms = max(0.0, random.gauss(30.0, 25.0))  # queue variance
        decode_ms = max(10.0, random.gauss(base_decode + 2.0 * (request.max_tokens or 128), 50.0))

        # Run stages
        await asyncio.sleep(retrieve_ms / 1000.0)
        await asyncio.sleep(rerank_ms / 1000.0)
        await asyncio.sleep(llm_queue_ms / 1000.0)
        await asyncio.sleep(decode_ms / 1000.0)

        # doc ids for overlap measurement
        # Make overlap more likely for short/medium
        pool = 200 if qc != "long" else 2000
        retrieved = [f"doc{random.randint(1, pool)}" for _ in range(k)]

        trace = rag_pb2.Trace(
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            llm_queue_ms=llm_queue_ms,
            decode_ms=decode_ms,
            k=k,
            prompt_tokens=500 + 20 * k,
            output_tokens=min(request.max_tokens or 128, 256),
            retrieved_doc_ids=retrieved,
        )
        return rag_pb2.RAGResponse(answer="stub answer", trace=trace)


async def serve(host: str = "0.0.0.0", port: int = 50051):
    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ])
    rag_pb2_grpc.add_RAGServiceServicer_to_server(RAGService(), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    print(f"gRPC server listening on {host}:{port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
