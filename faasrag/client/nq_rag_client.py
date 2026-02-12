import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import grpc
from tqdm import tqdm

import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc


@dataclass
class ClientConfig:
    target: str
    dataset_path: str
    limit: int
    shuffle: bool
    seed: int

    deadline_s: float
    concurrency: int
    retries: int
    retry_backoff_s: float

    out_jsonl: str
    flush_every: int


def trace_to_dict(trace: rag_pb2.Trace) -> Dict[str, Any]:
    return {
        "timings_s": dict(trace.timings_s),
        "cache_hits": int(trace.cache_hits),
        "cache_misses": int(trace.cache_misses),
        "cache_used": bool(trace.cache_used),
        "k": int(trace.k),
        "prompt_tokens": int(trace.prompt_tokens),
        "completion_tokens": int(trace.completion_tokens),
        "total_tokens": int(trace.total_tokens),
        "retrieved_doc_ids": list(trace.retrieved_doc_ids),
    }


def normalize_gold(g: Any) -> Optional[str]:
    """Turn golden_answers variants into a single representative string for logging."""
    if g is None:
        return None
    if isinstance(g, str):
        s = g.strip()
        return s or None
    if isinstance(g, list):
        # list of strings, pick the first non-empty
        for x in g:
            s = normalize_gold(x)
            if s:
                return s
        return None
    if isinstance(g, dict):
        # sometimes answers are dict-like
        for k in ("text", "answer", "value"):
            if k in g:
                s = normalize_gold(g[k])
                if s:
                    return s
    return str(g)


async def call_rag(
    stub: rag_pb2_grpc.RAGServiceStub,
    query: str,
    *,
    deadline_s: float,
    retries: int,
    retry_backoff_s: float,
) -> Tuple[Optional[rag_pb2.RAGResponse], Optional[str], float]:
    attempt = 0
    while True:
        attempt += 1
        t0 = time.perf_counter()
        try:
            req = rag_pb2.RAGRequest(query=query)
            resp = await stub.Query(req, timeout=deadline_s)
            return resp, None, (time.perf_counter() - t0)

        except grpc.aio.AioRpcError as e:
            latency = time.perf_counter() - t0
            code = e.code()
            msg = e.details() or str(e)
            err = f"{code.name}: {msg}"

            transient = code in {
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.INTERNAL,
            }
            if attempt <= retries and transient:
                await asyncio.sleep(retry_backoff_s * attempt)
                continue
            return None, err, latency

        except Exception as e:
            latency = time.perf_counter() - t0
            err = f"EXCEPTION: {type(e).__name__}: {e}"
            if attempt <= retries:
                await asyncio.sleep(retry_backoff_s * attempt)
                continue
            return None, err, latency


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and limit > 0 and i >= limit:
                break
            ex = json.loads(line)

            # Your stated contract: question + golden_answers
            if "question" not in ex or "golden_answers" not in ex:
                raise ValueError("Each jsonl line must contain fields: question, golden_answers")

            data.append(
                {
                    "question": ex["question"],
                    "golden_answers": ex["golden_answers"],
                    # keep any extra fields if you want:
                    # **ex
                }
            )
    return data


async def run(cfg: ClientConfig) -> None:
    qa = load_jsonl(cfg.dataset_path, limit=cfg.limit)

    if cfg.shuffle:
        rnd = random.Random(cfg.seed)
        rnd.shuffle(qa)

    channel = grpc.aio.insecure_channel(
        cfg.target,
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )
    stub = rag_pb2_grpc.RAGServiceStub(channel)
    sem = asyncio.Semaphore(cfg.concurrency)

    # line-buffered output so crashes lose less data
    out_f = open(cfg.out_jsonl, "w", encoding="utf-8", buffering=1)

    pbar = tqdm(total=len(qa), desc="NQ → RAG", unit="ex")
    written = 0

    async def handle_one(i: int, ex: Dict[str, Any]) -> None:
        nonlocal written

        q = ex.get("question")
        gold_raw = ex.get("golden_answers")

        record: Dict[str, Any] = {
            "idx": i,
            "question": q,
            "gold": normalize_gold(gold_raw),
            "gold_all": gold_raw,  # keep full list for evaluation later
            "dataset_path": cfg.dataset_path,
        }

        if not isinstance(q, str) or not q.strip():
            record["error"] = "NO_QUESTION_FOUND"
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            return

        async with sem:
            resp, err, latency_s = await call_rag(
                stub,
                q.strip(),
                deadline_s=cfg.deadline_s,
                retries=cfg.retries,
                retry_backoff_s=cfg.retry_backoff_s,
            )

        record["client_latency_s"] = float(latency_s)

        if err:
            record["error"] = err
        else:
            record["pred"] = resp.answer
            record["trace"] = trace_to_dict(resp.trace)

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        if cfg.flush_every > 0 and (written % cfg.flush_every == 0):
            out_f.flush()

    tasks: List[asyncio.Task] = []
    for i, ex in enumerate(qa):
        tasks.append(asyncio.create_task(handle_one(i, ex)))

        if len(tasks) >= cfg.concurrency * 4:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for _ in done:
                pbar.update(1)
            tasks = list(pending)

    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for _ in done:
            pbar.update(1)
        tasks = list(pending)

    pbar.close()
    out_f.close()
    await channel.close()


def parse_args() -> ClientConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, default="127.0.0.1:50051")
    ap.add_argument("--dataset_path", type=str, default="data/datasets/qa/nq/nq_dev.jsonl")
    ap.add_argument("--limit", type=int, default=20, help="0 = all")
    ap.add_argument("--shuffle", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--deadline_s", type=float, default=3000.0)

    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry_backoff_s", type=float, default=0.5)

    ap.add_argument("--out", type=str, default="nq_rag_results.jsonl")
    ap.add_argument("--flush_every", type=int, default=20)

    args = ap.parse_args()
    return ClientConfig(
        target=args.target,
        dataset_path=args.dataset_path,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        deadline_s=args.deadline_s,
        concurrency=args.concurrency,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
        out_jsonl=args.out,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    cfg = parse_args()
    asyncio.run(run(cfg))
