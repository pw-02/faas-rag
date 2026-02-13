from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt


"""
Chunk / Doc-ID reuse analysis for RAG retrieval.

Goal:
- Across a QA dataset, run retrieval for each query and measure how concentrated retrieval traffic is:
  - How many unique retrieved IDs (chunks/docs) are used at top-k?
  - What fraction of the corpus is that?
  - What fraction of total retrieval events are covered by the top X% most-frequent IDs?
  - Frequency distribution / heavy-tail evidence.

This supports the claim: caching text chunks (content objects) can be high-leverage because retrieval traffic is concentrated,
even if query-neighbor reuse is brittle.

Assumptions:
- Your FAISS doc index returns integer IDs that are stable across searches.
- If you built FAISS without IndexIDMap, IDs are 0..N-1 in add order.
"""


# -----------------------------
# IO / Embeddings
# -----------------------------
def load_questions_from_jsonl(path: str, max_q: Optional[int] = None) -> List[str]:
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question")
            if isinstance(q, str) and q.strip():
                qs.append(q.strip())
            if max_q is not None and len(qs) >= max_q:
                break
    return qs


@torch.no_grad()
def embed_questions(
    questions: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    out_chunks = []
    for s in range(0, len(questions), batch_size):
        batch = questions[s : s + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        out_chunks.append(emb)

    return np.vstack(out_chunks).astype(np.float32, copy=False)


def faiss_topk(index: faiss.Index, q: np.ndarray, k: int) -> np.ndarray:
    _, I = index.search(q.reshape(1, -1), k)
    return I[0]


# -----------------------------
# Analysis helpers
# -----------------------------
def get_faiss_ntotal(index: faiss.Index) -> Optional[int]:
    # Works for most indexes. If not available, returns None.
    try:
        return int(index.ntotal)
    except Exception:
        return None


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def gini_from_counts(counts: np.ndarray) -> float:
    """
    Gini coefficient for non-negative counts.
    0 = perfectly uniform, 1 = maximally concentrated.
    """
    counts = counts.astype(np.float64, copy=False)
    if counts.size == 0:
        return float("nan")
    if np.all(counts == 0):
        return 0.0
    counts = np.sort(counts)
    n = counts.size
    cum = np.cumsum(counts)
    # Gini = (n+1 - 2*sum_i (cum_i)/cum_n) / n
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def top_fraction_coverage(counts_sorted_desc: np.ndarray, frac: float) -> float:
    """
    Given counts sorted descending, compute fraction of total events covered by top frac of IDs.
    frac in (0,1], e.g. 0.01 = top 1% IDs.
    """
    n = counts_sorted_desc.size
    if n == 0:
        return float("nan")
    m = max(1, int(math.ceil(frac * n)))
    total = counts_sorted_desc.sum()
    if total <= 0:
        return float("nan")
    return float(counts_sorted_desc[:m].sum() / total)


def top_k_ids_coverage(counts_sorted_desc: np.ndarray, topn: int) -> float:
    if counts_sorted_desc.size == 0:
        return float("nan")
    topn = max(1, min(int(topn), int(counts_sorted_desc.size)))
    total = counts_sorted_desc.sum()
    if total <= 0:
        return float("nan")
    return float(counts_sorted_desc[:topn].sum() / total)


def save_summary_csv(summary: Dict[str, float], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])


def save_frequency_csv(counter: Counter, out_path: str) -> None:
    """
    Save per-id frequency counts.
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "count"])
        for doc_id, cnt in counter.most_common():
            w.writerow([int(doc_id), int(cnt)])


def save_curve_csv(
    counts_sorted_desc: np.ndarray,
    out_path: str,
    steps: int = 200,
) -> None:
    """
    Save "coverage curve": fraction of IDs vs fraction of events covered.
    We sample the curve at `steps` points (by number of IDs).
    """
    n = counts_sorted_desc.size
    total = counts_sorted_desc.sum()
    if n == 0 or total <= 0:
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id_frac", "event_coverage"])
        return

    cum = np.cumsum(counts_sorted_desc) / total  # coverage after top i IDs
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_frac", "event_coverage"])
        for t in range(1, steps + 1):
            i = int(round(t * n / steps))
            i = max(1, min(i, n))
            id_frac = i / n
            event_cov = float(cum[i - 1])
            w.writerow([id_frac, event_cov])


def plot_coverage_curve(curve_csv: str, out_png: str) -> None:
    df = pd.read_csv(curve_csv)
    if df.empty:
        raise SystemExit(f"No rows in {curve_csv}")
    plt.figure(figsize=(7, 5))
    plt.plot(df["id_frac"].values, df["event_coverage"].values, marker=None)
    plt.xlabel("Fraction of unique IDs cached (top by frequency)")
    plt.ylabel("Fraction of retrieval events covered")
    plt.title("Chunk/Doc reuse concentration: coverage curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")


def plot_loglog_rank_freq(freq_csv: str, out_png: str, max_points: int = 200000) -> None:
    """
    Zipf-like plot: rank vs frequency on log-log axes.
    """
    df = pd.read_csv(freq_csv)
    if df.empty:
        raise SystemExit(f"No rows in {freq_csv}")

    # already sorted in save_frequency_csv (most_common), but ensure:
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    if len(df) > max_points:
        df = df.iloc[:max_points].copy()

    ranks = np.arange(1, len(df) + 1, dtype=np.float64)
    freqs = df["count"].astype(np.float64).values

    plt.figure(figsize=(7, 5))
    plt.plot(ranks, freqs)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank of ID by retrieval frequency (log)")
    plt.ylabel("Frequency (log)")
    plt.title("Chunk/Doc reuse: rank-frequency (log-log)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")


# -----------------------------
# Main experiment
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data / models
    ap.add_argument("--questions_jsonl", default="data/datasets/qa/nq/nq_dev.jsonl")
    ap.add_argument("--dpr_q_model", default="sentence-transformers/facebook-dpr-question_encoder-single-nq-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_questions", type=int, default=50000)

    # retrieval
    ap.add_argument(
        "--doc_index",
        default="artifacts/wiki-dpr/faiss_wiki_dpr/hnsw_100k/index_psgs_w100_nq_no_index_hnsw_ip_100000.faiss",
        help="Main ANN index used for retrieval (simulates production index)",
    )
    ap.add_argument("--truth_index", default=None, help="Optional higher-accuracy index for retrieval (FlatIP or tuned ANN)")
    ap.add_argument("--use_truth_index", action="store_true", default=False, help="If set and --truth_index provided, use truth_index for retrieval instead of doc_index")
    ap.add_argument("--k", type=int, default=20, help="Top-k retrieved per query")

    # outputs
    ap.add_argument("--out_prefix", default="chunk_reuse")
    ap.add_argument("--make_plots", action="store_true", default=True)

    # misc
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load questions
    questions = load_questions_from_jsonl(args.questions_jsonl, max_q=args.max_questions)
    if len(questions) < 1:
        raise SystemExit("Need at least 1 question.")
    print(f"Loaded {len(questions)} questions from {args.questions_jsonl}")

    # Load indices
    doc_index = faiss.read_index(args.doc_index)
    truth_index = faiss.read_index(args.truth_index) if args.truth_index else None
    if args.use_truth_index and truth_index is None:
        raise SystemExit("--use_truth_index set but no --truth_index provided.")

    retrieval_index = truth_index if args.use_truth_index and truth_index is not None else doc_index
    retrieval_index_name = "truth_index" if retrieval_index is truth_index else "doc_index"
    print(f"Using {retrieval_index_name} for retrieval.")

    ntotal = get_faiss_ntotal(retrieval_index)
    if ntotal is not None:
        print(f"Index ntotal = {ntotal}")

    # Embed queries
    t0 = time.time()
    qvecs = embed_questions(
        questions=questions,
        model_name=args.dpr_q_model,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Embedded queries: shape={qvecs.shape} in {time.time()-t0:.1f}s")

    # Retrieve and count IDs
    k = int(args.k)
    total_queries = qvecs.shape[0]
    total_events = total_queries * k

    id_counter: Counter = Counter()

    t1 = time.time()
    for i in tqdm(range(total_queries), desc=f"retrieving top-{k}"):
        ids = faiss_topk(retrieval_index, qvecs[i], k)
        # FAISS can return -1 for missing entries in some configs; ignore those
        for doc_id in ids.tolist():
            if int(doc_id) >= 0:
                id_counter[int(doc_id)] += 1
    print(f"Retrieval done in {time.time()-t1:.1f}s")

    # Basic stats
    unique_ids_used = len(id_counter)
    counts = np.array(list(id_counter.values()), dtype=np.int64)
    counts_sorted_desc = np.sort(counts)[::-1]

    summary: Dict[str, float] = {}
    summary["num_queries"] = float(total_queries)
    summary["k"] = float(k)
    summary["total_retrieval_events"] = float(total_events)
    summary["unique_ids_used"] = float(unique_ids_used)
    summary["avg_reuse_per_id"] = safe_float(total_events / max(1, unique_ids_used))
    summary["median_count_per_id"] = safe_float(np.median(counts)) if counts.size else float("nan")
    summary["p90_count_per_id"] = safe_float(np.percentile(counts, 90)) if counts.size else float("nan")
    summary["p99_count_per_id"] = safe_float(np.percentile(counts, 99)) if counts.size else float("nan")
    summary["gini_count"] = gini_from_counts(counts)

    if ntotal is not None:
        summary["index_ntotal"] = float(ntotal)
        summary["fraction_of_corpus_touched"] = safe_float(unique_ids_used / max(1, ntotal))

    # Coverage by top fractions of IDs
    for frac in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        summary[f"event_coverage_top_{int(frac*1000)}permil_ids"] = top_fraction_coverage(counts_sorted_desc, frac)

    # Coverage by top-N IDs
    for topn in [10, 50, 100, 500, 1000, 5000]:
        summary[f"event_coverage_top_{topn}_ids"] = top_k_ids_coverage(counts_sorted_desc, topn)

    # Print a compact summary
    print("\n=== Chunk/Doc reuse summary ===")
    print(f"Queries: {total_queries}   k: {k}   total events: {total_events}")
    print(f"Unique IDs used: {unique_ids_used}   avg reuse/event per unique ID: {summary['avg_reuse_per_id']:.2f}")
    if ntotal is not None:
        print(f"Corpus touched: {unique_ids_used}/{ntotal} = {100.0*summary['fraction_of_corpus_touched']:.2f}%")
    print(f"Gini (concentration): {summary['gini_count']:.3f}")
    print("Event coverage by top IDs:")
    for frac in [0.01, 0.05, 0.10]:
        key = f"event_coverage_top_{int(frac*1000)}permil_ids"
        print(f"  Top {int(frac*100)}% IDs cover {100.0*summary[key]:.2f}% of events")
    for topn in [100, 1000, 5000]:
        key = f"event_coverage_top_{topn}_ids"
        print(f"  Top {topn} IDs cover {100.0*summary[key]:.2f}% of events")

    # Write outputs
    prefix = args.out_prefix
    out_summary_csv = f"{prefix}_summary.csv"
    out_freq_csv = f"{prefix}_freq.csv"
    out_curve_csv = f"{prefix}_coverage_curve.csv"

    save_summary_csv(summary, out_summary_csv)
    save_frequency_csv(id_counter, out_freq_csv)
    save_curve_csv(counts_sorted_desc, out_curve_csv, steps=250)

    print(f"\nSaved: {out_summary_csv}")
    print(f"Saved: {out_freq_csv}")
    print(f"Saved: {out_curve_csv}")

    # Plots
    if args.make_plots:
        plot_coverage_curve(out_curve_csv, f"{prefix}_coverage_curve.png")
        plot_loglog_rank_freq(out_freq_csv, f"{prefix}_rank_freq_loglog.png")


if __name__ == "__main__":
    main()
