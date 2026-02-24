#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np
import torch

from pwrag.retriever.caches import ProximityCache
from pwrag.retriever.encoder import STEncoder


# ----------------------------
# Cache factory
# ----------------------------

def make_cache(
    policy="fifo",
    tolerance=0.8,
    capacity=3,
    lsh_bucket_capacity=5,
    lsh_num_hashes=64,
    lsh_dim=8,
    lsh_seed=42,
) -> ProximityCache:
    return ProximityCache(
        policy=policy,
        tolerance=tolerance,
        capacity=capacity,
        lsh_bucket_capacity=lsh_bucket_capacity,
        lsh_num_hashes=lsh_num_hashes,
        lsh_dim=lsh_dim,
        lsh_seed=lsh_seed,
    )


# ----------------------------
# I/O
# ----------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} of {path}: {e}") from e


# ----------------------------
# Faiss wrapper
# ----------------------------

@dataclass
class FaissDB:
    index: faiss.Index
    id_map: Optional[np.ndarray]  # row -> doc_id

    @property
    def dim(self) -> int:
        return self.index.d

    def search_ids(self, query_embedding: np.ndarray, topk: int) -> np.ndarray:
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(f"Bad query shape {q.shape}, expected ({self.dim},)")
        _, I = self.index.search(q.reshape(1, -1), topk)
        ids = I[0].astype(np.int64)
        if self.id_map is not None:
            ids = np.where(ids >= 0, self.id_map[ids], -1)
        return ids


def load_faiss(index_path: Path, id_map_path: Optional[Path]) -> FaissDB:
    index = faiss.read_index(str(index_path))
    id_map = np.load(str(id_map_path)) if id_map_path else None
    if id_map is not None and id_map.ndim != 1:
        raise ValueError(f"id_map must be 1D, got {id_map.shape}")
    return FaissDB(index=index, id_map=id_map)


# ----------------------------
# Cache adapter (your API)
# ----------------------------

def cache_lookup(cache: ProximityCache, query_embedding: np.ndarray) -> Optional[List[int]]:
    return cache.find(query_embedding)

def cache_store(cache: ProximityCache, query_embedding: np.ndarray, retrieved_doc_ids: List[int]) -> None:
    cache.insert(query_embedding, retrieved_doc_ids)


# ----------------------------
# Timing helper
# ----------------------------

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0


# ----------------------------
# Metrics / summary
# ----------------------------

@dataclass
class Summary:
    dataset: str
    num_queries: int
    hit_rate: float
    avg_overlap: float
    wall_s: float
    qps_measured: float
    qps_effective_no_gt: float
    avg_total_s: float
    avg_gt_s: float
    avg_cache_s: float
    avg_retrieve_s: float
    avg_encode_s: float


def eval_one_dataset(
    db: FaissDB,
    cache: ProximityCache,
    encoder: STEncoder,
    dataset_path: Path,
    topk_gt: int,
    topk_ret: int,
    encode_batch: int,
    details_csv: Optional[Path],
) -> Summary:
    """
    Clean + efficient behavior:

    - Encode questions in batches.
    - For each query:
        1) cache lookup
        2a) HIT: retrieved_doc_ids = cached; run a FAISS search only for GT (eval-only)
        2b) MISS: run ONE FAISS search with k=max(topk_gt, topk_ret)
                - GT = first topk_gt
                - Retrieved = first topk_ret
                - insert Retrieved into cache
    - Overlap = |Retrieved ∩ GT|
    - "effective_no_gt" subtracts ONLY the eval-only GT time (which happens only on cache hits).
    """

    # Optional details writer
    details_f = None
    details_writer = None
    if details_csv is not None:
        details_csv.parent.mkdir(parents=True, exist_ok=True)
        details_f = details_csv.open("w", newline="", encoding="utf-8")
        details_writer = csv.DictWriter(
            details_f,
            fieldnames=[
                "id", "cache_hit", "overlap",
                "t_total", "t_encode", "t_gt", "t_cache", "t_retrieve",
                "t_effective_no_gt",
            ],
        )
        details_writer.writeheader()

    # Aggregate counters/timers
    num_queries = 0
    num_cache_hits = 0
    overlap_sum = 0

    sum_total = 0.0
    sum_encode = 0.0
    sum_gt_eval_only = 0.0
    sum_cache = 0.0
    sum_retrieve = 0.0

    # Batch buffers (only what we actually use)
    batch_query_ids: List[str] = []
    batch_questions: List[str] = []

    def process_batch() -> None:
        nonlocal num_queries, num_cache_hits, overlap_sum
        nonlocal sum_total, sum_encode, sum_gt_eval_only, sum_cache, sum_retrieve

        if not batch_questions:
            return

        # Encode batch (amortize per query for reporting)
        with Timer() as t_encode:
            question_embeddings = encoder.encode(batch_questions)  # type: ignore[attr-defined]
        question_embeddings = np.asarray(question_embeddings, dtype=np.float32)

        if question_embeddings.ndim != 2:
            raise ValueError(f"Encoder returned shape {question_embeddings.shape}, expected (B, D)")
        if question_embeddings.shape[1] != db.dim:
            raise ValueError(f"Encoder dim {question_embeddings.shape[1]} != Faiss dim {db.dim}")

        per_query_encode = t_encode.dt / len(batch_questions)

        for qid, query_embedding in zip(batch_query_ids, question_embeddings):
            num_queries += 1

            with Timer() as t_total:
                # 1) Cache lookup
                with Timer() as t_cache:
                    cached_doc_ids = cache_lookup(cache, query_embedding)

                # 2) Retrieval + GT
                if cached_doc_ids is not None:
                    # HIT: use cached docs for retrieval; do GT search only for eval
                    cache_hit = True
                    retrieved_doc_ids = cached_doc_ids
                    retrieve_time = 0.0

                    with Timer() as t_gt:
                        gt_doc_ids = db.search_ids(query_embedding, topk_gt)
                    gt_eval_time = t_gt.dt

                else:
                    # MISS: single FAISS search reused for both GT + retrieval
                    cache_hit = False
                    k = max(topk_gt, topk_ret)

                    with Timer() as t_retrieve:
                        all_doc_ids = db.search_ids(query_embedding, k)
                    retrieve_time = t_retrieve.dt
                    gt_eval_time = 0.0  # no extra "GT-only" search on misses

                    gt_doc_ids = all_doc_ids[:topk_gt]
                    retrieved_ids_np = all_doc_ids[:topk_ret]
                    retrieved_doc_ids = [int(x) for x in retrieved_ids_np.tolist() if int(x) >= 0]

                    cache_store(cache, query_embedding, retrieved_doc_ids)

                # 3) Overlap
                gt_set = {int(x) for x in gt_doc_ids.tolist() if int(x) >= 0}
                overlap = len(set(retrieved_doc_ids) & gt_set)

            if cache_hit:
                num_cache_hits += 1
            overlap_sum += overlap

            sum_total += t_total.dt
            sum_encode += per_query_encode
            sum_gt_eval_only += gt_eval_time
            sum_cache += t_cache.dt
            sum_retrieve += retrieve_time

            if details_writer is not None:
                details_writer.writerow(
                    {
                        "id": qid,
                        "cache_hit": int(cache_hit),
                        "overlap": overlap,
                        "t_total": f"{t_total.dt:.6f}",
                        "t_encode": f"{per_query_encode:.6f}",
                        "t_gt": f"{gt_eval_time:.6f}",
                        "t_cache": f"{t_cache.dt:.6f}",
                        "t_retrieve": f"{retrieve_time:.6f}",
                        "t_effective_no_gt": f"{max(0.0, t_total.dt - gt_eval_time):.6f}",
                    }
                )

        batch_query_ids.clear()
        batch_questions.clear()

    # Stream dataset, encode in batches
    wall_start = time.perf_counter()

    for obj in read_jsonl(dataset_path):
        qid = str(obj.get("id", f"row_{len(batch_query_ids)+1}"))
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing/invalid 'question' for id={qid}")

        batch_query_ids.append(qid)
        batch_questions.append(question)

        if len(batch_questions) >= encode_batch:
            process_batch()

    process_batch()
    wall_time = time.perf_counter() - wall_start

    if details_f is not None:
        details_f.close()

    hit_rate = (num_cache_hits / num_queries) if num_queries else 0.0
    avg_overlap = (overlap_sum / num_queries) if num_queries else 0.0

    qps_measured = (num_queries / wall_time) if wall_time > 0 else 0.0
    effective_wall = max(1e-9, wall_time - sum_gt_eval_only)  # subtract eval-only GT time
    qps_effective = num_queries / effective_wall

    return Summary(
        dataset=str(dataset_path),
        num_queries=num_queries,
        hit_rate=hit_rate,
        avg_overlap=avg_overlap,
        wall_s=wall_time,
        qps_measured=qps_measured,
        qps_effective_no_gt=qps_effective,
        avg_total_s=(sum_total / num_queries) if num_queries else 0.0,
        avg_gt_s=(sum_gt_eval_only / num_queries) if num_queries else 0.0,
        avg_cache_s=(sum_cache / num_queries) if num_queries else 0.0,
        avg_retrieve_s=(sum_retrieve / num_queries) if num_queries else 0.0,
        avg_encode_s=(sum_encode / num_queries) if num_queries else 0.0,
    )


def write_summary_csv(path: Path, summaries: List[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "num_queries", "hit_rate", "avg_overlap",
                "wall_s", "qps_measured", "qps_effective_no_gt",
                "avg_total_s", "avg_encode_s", "avg_gt_s", "avg_cache_s", "avg_retrieve_s",
            ],
        )
        w.writeheader()
        for s in summaries:
            w.writerow(s.__dict__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="corpus/faiss_wiki_dpr/flat_100k/index_psgs_w100_nq_no_index_flat_ip_100000.faiss")
    ap.add_argument("--id_map", default=None)
    ap.add_argument("--datasets", nargs="+", default=
                     [ 
                        # "data/datasets/qa/nq/nq_dev.jsonl",
                        "data/datasets/mmlu/subjects/econometrics/dev.jsonl",
                    ])

    ap.add_argument("--topk_gt", type=int, default=5)
    ap.add_argument("--topk_ret", type=int, default=5)
    ap.add_argument("--encode_batch", type=int, default=1)

    ap.add_argument("--out_dir", default="results/cache_eval")
    ap.add_argument("--details", action="store_true", default=False)

    # cache params
    ap.add_argument("--policy", default="fifo")
    ap.add_argument("--tolerance", type=float, default=0.8)
    ap.add_argument("--capacity", type=int, default=1000)
    ap.add_argument("--lsh_bucket_capacity", type=int, default=5)
    ap.add_argument("--lsh_num_hashes", type=int, default=64)
    ap.add_argument("--lsh_dim", type=int, default=8)
    ap.add_argument("--lsh_seed", type=int, default=42)

    # encoder params (make explicit so you can swap models without editing code)
    ap.add_argument("--encoder_model_name", default="dpr")
    ap.add_argument("--encoder_model_path", default="facebook/dpr-question_encoder-single-nq-base")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()

    db = load_faiss(Path(args.index), Path(args.id_map) if args.id_map else None)
    cache = make_cache(
        policy=args.policy,
        tolerance=args.tolerance,
        capacity=args.capacity,
        lsh_bucket_capacity=args.lsh_bucket_capacity,
        lsh_num_hashes=args.lsh_num_hashes,
        lsh_dim=args.lsh_dim,
        lsh_seed=args.lsh_seed,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoder = STEncoder(
        model_name=args.encoder_model_name,
        model_path=args.encoder_model_path,
        use_fp16=args.fp16,
        max_length=args.max_length,
        instruction=None,
        device=device,
        silent=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Summary] = []
    for dataset in args.datasets:
        dataset_path = Path(dataset)
        details_csv = (out_dir / f"{dataset_path.stem}.details.csv") if args.details else None

        summary = eval_one_dataset(
            db=db,
            cache=cache,
            encoder=encoder,
            dataset_path=dataset_path,
            topk_gt=args.topk_gt,
            topk_ret=args.topk_ret,
            encode_batch=args.encode_batch,
            details_csv=details_csv,
        )
        summaries.append(summary)

        print(
            f"{dataset_path.name}: hit_rate={summary.hit_rate:.3f}, "
            f"avg_overlap={summary.avg_overlap:.3f}, "
            f"qps(measured)={summary.qps_measured:.2f}, "
            f"qps(no_gt)={summary.qps_effective_no_gt:.2f}"
        )

    write_summary_csv(out_dir / "summary.csv", summaries)
    print(f"Wrote: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()