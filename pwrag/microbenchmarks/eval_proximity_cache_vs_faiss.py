#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import faiss
import numpy as np
import torch

from pwrag.retriever.caches import ProximityCache
from pwrag.retriever.encoder import STEncoder


# ----------------------------
# I/O
# ----------------------------
def read_jsonl_questions(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSON on line {line_no} of {path}: {e}") from e
            q = obj.get("question")
            if not isinstance(q, str) or not q.strip():
                raise ValueError(f"Missing/invalid 'question' on line {line_no} of {path}")
            yield q


# ----------------------------
# FAISS wrapper
# ----------------------------
@dataclass
class FaissDB:
    index: faiss.Index
    id_map: Optional[np.ndarray]  # row -> doc_id

    @property
    def dim(self) -> int:
        return self.index.d

    def search_ids(self, query_embedding: np.ndarray, topk: int) -> List[int]:
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(f"Bad query shape {q.shape}, expected ({self.dim},)")
        _, I = self.index.search(q.reshape(1, -1), topk)
        ids = I[0].astype(np.int64)
        if self.id_map is not None:
            ids = np.where(ids >= 0, self.id_map[ids], -1)
        return [int(x) for x in ids.tolist() if int(x) >= 0]


def load_faiss(index_path: Path, id_map_path: Optional[Path]) -> FaissDB:
    index = faiss.read_index(str(index_path))
    id_map = np.load(str(id_map_path)) if id_map_path else None
    if id_map is not None and id_map.ndim != 1:
        raise ValueError(f"id_map must be 1D, got {id_map.shape}")
    return FaissDB(index=index, id_map=id_map)


# ----------------------------
# Cache factory
# ----------------------------
def make_cache(
    policy: str,
    tolerance: float,
    capacity: int,
    lsh_bucket_capacity: int,
    lsh_num_hashes: int,
    lsh_dim: int,
    lsh_seed: int,
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
# Metrics
# ----------------------------
@dataclass
class Row:
    dataset: str
    tolerance: float
    topk: int
    num_queries: int
    num_hits: int
    hit_rate: float
    avg_overlap_on_hits: float           # in [0..topk]
    avg_overlap_pct_on_hits: float       # in [0..1]


def overlap_count(a: List[int], b: List[int]) -> int:
    return len(set(a) & set(b))


def eval_dataset_for_tolerance(
    *,
    db: FaissDB,
    encoder: STEncoder,
    dataset_path: Path,
    tolerance: float,
    topk: int,
    policy: str,
    encode_batch: int,
    capacity: int,
    lsh_bucket_capacity: int,
    lsh_num_hashes: int,
    lsh_dim: int,
    lsh_seed: int,
) -> Row:
    cache = make_cache(
        policy=policy,
        tolerance=tolerance,
        capacity=capacity,
        lsh_bucket_capacity=lsh_bucket_capacity,
        lsh_num_hashes=lsh_num_hashes,
        lsh_dim=lsh_dim,
        lsh_seed=lsh_seed,
    )

    num_queries = 0
    num_hits = 0
    overlap_sum_hits = 0

    # batching just for encoder efficiency; logic stays simple
    batch_q: List[str] = []

    def process_batch(questions: List[str]) -> None:
        nonlocal num_queries, num_hits, overlap_sum_hits

        if not questions:
            return

        embs = encoder.encode(questions)  # type: ignore[attr-defined]
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim != 2 or embs.shape[1] != db.dim:
            raise ValueError(f"Encoder returned {embs.shape}, expected (B, {db.dim})")

        for emb in embs:
            num_queries += 1

            cached = cache.find(emb)

            # Define baseline as "what FAISS would return now"
            baseline = db.search_ids(emb, topk)

            if cached is not None:
                # HIT: measure hit-accuracy by comparing cached results to baseline
                num_hits += 1
                ov = overlap_count(cached[:topk], baseline)
                overlap_sum_hits += ov
            else:
                # MISS: store the baseline retrieval (or topk retrieval) into cache
                cache.insert(emb, baseline)

    for q in read_jsonl_questions(dataset_path):
        batch_q.append(q)
        if len(batch_q) >= encode_batch:
            process_batch(batch_q)
            batch_q.clear()

    if batch_q:
        process_batch(batch_q)

    hit_rate = (num_hits / num_queries) if num_queries else 0.0
    avg_overlap_on_hits = (overlap_sum_hits / num_hits) if num_hits else 0.0
    avg_overlap_pct_on_hits = (avg_overlap_on_hits / topk) if topk > 0 else 0.0

    return Row(
        dataset=str(dataset_path),
        tolerance=float(tolerance),
        topk=int(topk),
        num_queries=int(num_queries),
        num_hits=int(num_hits),
        hit_rate=float(hit_rate),
        #Average number of retrieved documents (out of topk) that match what FAISS would have returned when the cache hit
        #i.e, How many of the cached docs match FAISS docs on average?
        avg_overlap_on_hits=float(avg_overlap_on_hits), #
        # When the cache is used instead of FAISS, how similar are the returned documents to what FAISS would have returned?
        avg_overlap_pct_on_hits=float(avg_overlap_pct_on_hits)
    )


def append_rows_csv(csv_path: Path, rows: List[Row]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def collect_datasets(folders: List[str]) -> List[str]:
    out: List[str] = []
    for folder in folders:
        p = Path(folder)
        if p.is_dir():
            out.extend(str(x) for x in p.glob("**/*.jsonl"))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="corpus/faiss_wiki_dpr/hnsw_all/index_psgs_w100_nq_no_index_hnsw_ip_all.faiss")
    ap.add_argument("--id_map", default=None)

    ap.add_argument("--dataset_folders", nargs="+", default=[])
    
    ap.add_argument("--datasets", nargs="+", default=[
    "data/datasets/mmlu/mmlu/mmlu_dev.jsonl",
    "data/datasets/mmlu/mmlu/mmlu_test.jsonl",
    "data/datasets/mmlu/mmlu/mmlu_train.jsonl",
    "data/datasets/qa/nq/nq_dev.jsonl",
    "data/datasets/qa/nq/nq_test.jsonl",
    "data/datasets/qa/nq/nq_train.jsonl",
    "data/datasets/qa/squad/squad_dev.jsonl",
    "data/datasets/qa/squad/squad_train.jsonl",
    "data/datasets/qa/triviaqa/triviaqa_dev.jsonl",
    "data/datasets/qa/triviaqa/triviaqa_test.jsonl",
    "data/datasets/qa/triviaqa/triviaqa_train.jsonl",
    "data/datasets/qa/wikiqa/wikiqa_dev.jsonl",
    "data/datasets/qa/wikiqa/wikiqa_test.jsonl",
    "data/datasets/qa/wikiqa/wikiqa_train.jsonl"
    ], help="Optional explicit list of jsonl paths; overrides dataset_folders")

    ap.add_argument("--out_csv", default="results/cache_eval/proximity_eval.csv")

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--encode_batch", type=int, default=16)

    # cache params
    ap.add_argument("--policy", default="fifo")
    ap.add_argument("--tolerances", nargs="+", type=float, default=[0, 2, 4, 6, 8, 10])

    ap.add_argument("--capacity", type=int, default=100000)  # can override per dataset if you want
    ap.add_argument("--capacity_use_dataset_size", action="store_true", default=True)

    ap.add_argument("--lsh_bucket_capacity", type=int, default=5)
    ap.add_argument("--lsh_num_hashes", type=int, default=64)
    ap.add_argument("--lsh_dim", type=int, default=8)
    ap.add_argument("--lsh_seed", type=int, default=42)

    # encoder params
    ap.add_argument("--encoder_model_name", default="dpr")
    ap.add_argument("--encoder_model_path", default="facebook/dpr-question_encoder-single-nq-base")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()

    datasets = args.datasets if args.datasets else collect_datasets(args.dataset_folders)
    if not datasets:
        raise SystemExit("No datasets found.")

    db = load_faiss(Path(args.index), Path(args.id_map) if args.id_map else None)

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

    out_csv = Path(args.out_csv)
    all_rows: List[Row] = []

    for dataset in datasets:
        dataset_path = Path(dataset)

        # optionally size capacity to dataset length (so you don’t evict)
        if args.capacity_use_dataset_size:
            with dataset_path.open("r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            capacity = max(1, n)
        else:
            capacity = args.capacity

        for tol in args.tolerances:
            row = eval_dataset_for_tolerance(
                db=db,
                encoder=encoder,
                dataset_path=dataset_path,
                tolerance=tol,
                topk=args.topk,
                policy=args.policy,
                encode_batch=args.encode_batch,
                capacity=capacity,
                lsh_bucket_capacity=args.lsh_bucket_capacity,
                lsh_num_hashes=args.lsh_num_hashes,
                lsh_dim=args.lsh_dim,
                lsh_seed=args.lsh_seed,
            )
            all_rows.append(row)

            print(
                f"{dataset_path} | tol={tol:g} | "
                f"hits={row.num_hits}/{row.num_queries} ({row.hit_rate*100:.1f}%) | "
                f"hit-accuracy(overlap@{args.topk})={row.avg_overlap_pct_on_hits*100:.1f}%"
            )

    append_rows_csv(out_csv, all_rows)
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()