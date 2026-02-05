import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import grpc
import numpy as np
import pandas as pd
from tqdm import tqdm

import rag_pb2
import rag_pb2_grpc


@dataclass
class Result:
    request_id: int
    ok: bool
    error: str
    start_ts: float
    end_ts: float
    e2e_ms: float

    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_queue_ms: float = 0.0
    decode_ms: float = 0.0

    k: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    retrieved_doc_ids: Optional[List[str]] = None


def load_queries(path: str) -> List[Dict[str, Any]]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    if not qs:
        raise ValueError("Empty queries file")
    return qs


def sample_query(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return random.choice(queries)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values), p))


def compute_overlap(results: List[Result], window_s: float) -> float:
    rows = [(r.start_ts, r.retrieved_doc_ids) for r in results if r.ok and r.retrieved_doc_ids]
    if len(rows) < 2:
        return float("nan")

    rows.sort(key=lambda x: x[0])
    seen: List[Tuple[float, set]] = []
    dup = 0
    total = 0

    for ts, ids in rows:
        cur = set(ids)
        seen = [(t, s) for (t, s) in seen if ts - t <= window_s]
        prev_union = set().union(*(s for _, s in seen)) if seen else set()

        total += len(cur)
        dup += len(cur.intersection(prev_union))
        seen.append((ts, cur))

    return dup / total if total else float("nan")


def summarize(results: List[Result]) -> Dict[str, Any]:
    oks = [r for r in results if r.ok]
    e2e = [r.e2e_ms for r in oks]
    out = {
        "n_total": len(results),
        "n_ok": len(oks),
        "success_rate": (len(oks) / len(results)) if results else 0.0,
        "throughput_rps": (len(oks) / (max(r.end_ts for r in oks) - min(r.start_ts for r in oks))) if len(oks) > 1 else 0.0,
        "lat_p50_ms": percentile(e2e, 50),
        "lat_p95_ms": percentile(e2e, 95),
        "lat_p99_ms": percentile(e2e, 99),
        "retrieve_p50_ms": percentile([r.retrieve_ms for r in oks], 50) if oks else float("nan"),
        "rerank_p50_ms": percentile([r.rerank_ms for r in oks], 50) if oks else float("nan"),
        "llm_queue_p50_ms": percentile([r.llm_queue_ms for r in oks], 50) if oks else float("nan"),
        "decode_p50_ms": percentile([r.decode_ms for r in oks], 50) if oks else float("nan"),
        "dup_ratio_1s": compute_overlap(results, 1.0),
        "dup_ratio_5s": compute_overlap(results, 5.0),
        "dup_ratio_30s": compute_overlap(results, 30.0),
    }
    return out


async def send_one(stub, req_id: int, q: Dict[str, Any], timeout_s: float) -> Result:
    t0 = time.time()
    try:
        req = rag_pb2.RAGRequest(
            query=q["query"],
            k=int(q.get("k", 0) or 0),
            max_tokens=int(q.get("max_tokens", 0) or 0),
            query_class=str(q.get("class", "")),
        )
        resp: rag_pb2.RAGResponse = await stub.Query(req, timeout=timeout_s)
        t1 = time.time()

        tr = resp.trace
        return Result(
            request_id=req_id,
            ok=True,
            error="",
            start_ts=t0,
            end_ts=t1,
            e2e_ms=(t1 - t0) * 1000.0,
            retrieve_ms=float(tr.retrieve_ms),
            rerank_ms=float(tr.rerank_ms),
            llm_queue_ms=float(tr.llm_queue_ms),
            decode_ms=float(tr.decode_ms),
            k=int(tr.k),
            prompt_tokens=int(tr.prompt_tokens),
            output_tokens=int(tr.output_tokens),
            retrieved_doc_ids=list(tr.retrieved_doc_ids),
        )
    except Exception as e:
        t1 = time.time()
        return Result(
            request_id=req_id,
            ok=False,
            error=str(e),
            start_ts=t0,
            end_ts=t1,
            e2e_ms=(t1 - t0) * 1000.0,
        )


async def run_fixed_concurrency(target: str, queries: List[Dict[str, Any]], concurrency: int, total: int, timeout_s: float, seed: int):
    random.seed(seed)
    results: List[Result] = []
    sem = asyncio.Semaphore(concurrency)

    async with grpc.aio.insecure_channel(target) as channel:
        stub = rag_pb2_grpc.RAGServiceStub(channel)

        async def worker(req_id: int):
            q = sample_query(queries)
            async with sem:
                r = await send_one(stub, req_id, q, timeout_s)
                results.append(r)

        tasks = [asyncio.create_task(worker(i)) for i in range(total)]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"conc={concurrency}"):
            await f

    results.sort(key=lambda r: r.request_id)
    return results


async def run_fixed_qps(target: str, queries: List[Dict[str, Any]], qps: float, duration_s: float, max_inflight: int, timeout_s: float, seed: int):
    random.seed(seed)
    results: List[Result] = []
    inflight = set()

    async with grpc.aio.insecure_channel(target) as channel:
        stub = rag_pb2_grpc.RAGServiceStub(channel)

        async def launch(req_id: int):
            q = sample_query(queries)
            r = await send_one(stub, req_id, q, timeout_s)
            results.append(r)

        t_end = time.time() + duration_s
        req_id = 0
        interval = 1.0 / qps if qps > 0 else 0.0

        while time.time() < t_end:
            while len(inflight) >= max_inflight:
                done, inflight = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
            task = asyncio.create_task(launch(req_id))
            inflight.add(task)
            req_id += 1
            await asyncio.sleep(interval)

        if inflight:
            await asyncio.wait(inflight)

    results.sort(key=lambda r: r.request_id)
    return results


async def run_bursty(target: str, queries: List[Dict[str, Any]], burst_qps: float, on_s: float, off_s: float, cycles: int, max_inflight: int, timeout_s: float, seed: int):
    random.seed(seed)
    results: List[Result] = []
    rid = 0

    for c in range(cycles):
        on_res = await run_fixed_qps(
            target=target,
            queries=queries,
            qps=burst_qps,
            duration_s=on_s,
            max_inflight=max_inflight,
            timeout_s=timeout_s,
            seed=seed + c,
        )
        for r in on_res:
            r.request_id = rid
            rid += 1
        results.extend(on_res)
        await asyncio.sleep(off_s)

    results.sort(key=lambda r: r.request_id)
    return results


def save(results: List[Result], out_csv: str):
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="host:port, e.g., localhost:50051")
    ap.add_argument("--queries", required=True, help="queries JSONL")
    ap.add_argument("--outdir", default="out_grpc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0)

    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("concurrency")
    c.add_argument("--concurrency", type=int, required=True)
    c.add_argument("--total_requests", type=int, default=500)

    q = sub.add_parser("qps")
    q.add_argument("--qps", type=float, required=True)
    q.add_argument("--duration_s", type=float, default=60.0)
    q.add_argument("--max_inflight", type=int, default=200)

    b = sub.add_parser("bursty")
    b.add_argument("--burst_qps", type=float, required=True)
    b.add_argument("--on_s", type=float, default=10.0)
    b.add_argument("--off_s", type=float, default=10.0)
    b.add_argument("--cycles", type=int, default=10)
    b.add_argument("--max_inflight", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    queries = load_queries(args.queries)

    if args.mode == "concurrency":
        results = asyncio.run(run_fixed_concurrency(args.target, queries, args.concurrency, args.total_requests, args.timeout, args.seed))
        tag = f"conc{args.concurrency}_n{args.total_requests}"
    elif args.mode == "qps":
        results = asyncio.run(run_fixed_qps(args.target, queries, args.qps, args.duration_s, args.max_inflight, args.timeout, args.seed))
        tag = f"qps{args.qps}_dur{args.duration_s}"
    else:
        results = asyncio.run(run_bursty(args.target, queries, args.burst_qps, args.on_s, args.off_s, args.cycles, args.max_inflight, args.timeout, args.seed))
        tag = f"bursty{args.burst_qps}_on{args.on_s}_off{args.off_s}_cyc{args.cycles}"

    out_csv = os.path.join(args.outdir, f"results_{tag}.csv")
    out_json = os.path.join(args.outdir, f"summary_{tag}.json")
    save(results, out_csv)
    summ = summarize(results)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summ.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
    print(f"\nSaved results: {out_csv}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
