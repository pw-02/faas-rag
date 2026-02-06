
import argparse
import sys
from typing import Any

import grpc

import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--query", default="Who wrote The Hobbit?")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--timeout_s", type=float, default=300000.0)
    args = ap.parse_args()

    target = f"{args.host}:{args.port}"
    print(f"Connecting to {target}")

    channel = grpc.insecure_channel(target)
    stub = rag_pb2_grpc.RAGServiceStub(channel)

    req = rag_pb2.RAGRequest(
        query=args.query,
        k=args.k,
        max_tokens=args.max_tokens,
    )

    try:
        resp: rag_pb2.RAGResponse = stub.Query(req, timeout=args.timeout_s)
    except grpc.RpcError as e:
        print("RPC failed!")
        print(f"  code   = {e.code()}")
        print(f"  detail = {e.details()}")
        sys.exit(1)

    print("\n=== ANSWER ===")
    print(resp.answer)

    if not resp.HasField("trace"):
        print("\n(No trace returned)")
        return

    trace = resp.trace

    print("\n=== TRACE ===")
    print(f"k={trace.k} prompt_tokens={trace.prompt_tokens} output_tokens={trace.output_tokens}")
    print(f"cache_used={trace.cache_used} hits={trace.cache_hits} misses={trace.cache_misses}")

    if trace.retrieved_doc_ids:
        print("\nretrieved_doc_ids:")
        for d in trace.retrieved_doc_ids[:20]:
            print(f"  - {d}")
        if len(trace.retrieved_doc_ids) > 20:
            print(f"  ... ({len(trace.retrieved_doc_ids)} total)")

    if trace.timings_s:
        print("\ntimings_s (seconds):")
        # sort by key for stable output
        for k, v in sorted(trace.timings_s.items(), key=lambda kv: kv[0]):
            print(f"  {k:20s} {float(v):.6f}")

    channel.close()


if __name__ == "__main__":
    main()
