from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt


"""
Chunk / Doc-ID reuse analysis for RAG retrieval.

UPDATED: supports multiple datasets and writes outputs under:
  <out_dir>/<dataset_name>/
    chunk_reuse_summary.csv
    chunk_reuse_freq.csv
    chunk_reuse_coverage_curve.csv
    chunk_reuse_coverage_curve.png
    chunk_reuse_rank_freq_loglog.png
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
def dataset_name_from_path(path: str) -> str:
    """
    Make a safe directory name from the dataset filename.
    Examples:
      /a/b/nq_dev.jsonl -> nq_dev
      triviaqa.jsonl    -> triviaqa
    """
    base = os.path.basename(path)
    name = re.sub(r"\.jsonl$", "", base, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "dataset"


def get_faiss_ntotal(index: faiss.Index) -> Optional[int]:
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
    return float((n + 1 - 2.0 * np.sum(cum) / cum[-1]) / n)


def top_fraction_coverage(counts_sorted_desc: np.ndarray, frac: float) -> float:
    """
    Fraction of total events covered by top frac of IDs.
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
    Coverage curve: fraction of IDs vs fraction of events covered.
    """
    n = counts_sorted_desc.size
    total = counts_sorted_desc.sum()
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_frac", "event_coverage"])
        if n == 0 or total <= 0:
            return

        cum = np.cumsum(counts_sorted_desc) / total
        for t in range(1, steps + 1):
            i = int(round(t * n / steps))
            i = max(1, min(i, n))
            id_frac = i / n
            event_cov = float(cum[i - 1])
            w.writerow([id_frac, event_cov])


def plot_coverage_curve(curve_csv: str, out_png: str) -> None:
    df = pd.read_csv(curve_csv)
    if df.empty:
        print(f"Skipping plot (no rows): {curve_csv}")
        return
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
    df = pd.read_csv(freq_csv)
    if df.empty:
        print(f"Skipping plot (no rows): {freq_csv}")
        return

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
# Per-dataset run
# -----------------------------
def run_one_dataset(
    *,
    questions_jsonl: str,
    retrieval_index: faiss.Index,
    ntotal: Optional[int],
    dpr_q_model: str,
    batch_size: int,
    device: str,
    max_questions: int,
    k: int,
    out_dir: str,
    make_plots: bool,
) -> None:
    dname = dataset_name_from_path(questions_jsonl)
    ds_out = os.path.join(out_dir, dname)
    os.makedirs(ds_out, exist_ok=True)

    out_summary_csv = os.path.join(ds_out, "chunk_reuse_summary.csv")
    out_freq_csv = os.path.join(ds_out, "chunk_reuse_freq.csv")
    out_curve_csv = os.path.join(ds_out, "chunk_reuse_coverage_curve.csv")
    out_curve_png = os.path.join(ds_out, "chunk_reuse_coverage_curve.png")
    out_rank_png = os.path.join(ds_out, "chunk_reuse_rank_freq_loglog.png")

    print("\n" + "=" * 90)
    print(f"DATASET: {dname}")
    print(f"INPUT : {questions_jsonl}")
    print(f"OUTDIR: {ds_out}")
    print("=" * 90)

    # Load questions
    questions = load_questions_from_jsonl(questions_jsonl, max_q=max_questions)
    if len(questions) < 1:
        print(f"Skipping {questions_jsonl}: no questions.")
        return
    print(f"Loaded {len(questions)} questions")

    # Embed queries
    t0 = time.time()
    qvecs = embed_questions(
        questions=questions,
        model_name=dpr_q_model,
        batch_size=batch_size,
        device=device,
    )
    print(f"Embedded queries: shape={qvecs.shape} in {time.time()-t0:.1f}s")

    # Retrieve and count IDs
    total_queries = qvecs.shape[0]
    total_events = total_queries * k
    id_counter: Counter = Counter()

    t1 = time.time()
    for i in tqdm(range(total_queries), desc=f"{dname}: retrieving top-{k}"):
        ids = faiss_topk(retrieval_index, qvecs[i], k)
        for doc_id in ids.tolist():
            if int(doc_id) >= 0:
                id_counter[int(doc_id)] += 1
    print(f"Retrieval done in {time.time()-t1:.1f}s")

    # Basic stats
    unique_ids_used = len(id_counter)
    counts = np.array(list(id_counter.values()), dtype=np.int64)
    counts_sorted_desc = np.sort(counts)[::-1]

    summary: Dict[str, float] = {}
    summary["dataset"] = dname  # useful for quick greps
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

    # Coverage by top fractions of IDs (permil = per thousand)
    for frac in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        summary[f"event_coverage_top_{int(frac*1000)}permil_ids"] = top_fraction_coverage(counts_sorted_desc, frac)

    # Coverage by top-N IDs
    for topn in [10, 50, 100, 500, 1000, 5000]:
        summary[f"event_coverage_top_{topn}_ids"] = top_k_ids_coverage(counts_sorted_desc, topn)

    # Write outputs
    save_summary_csv(summary, out_summary_csv)
    save_frequency_csv(id_counter, out_freq_csv)
    save_curve_csv(counts_sorted_desc, out_curve_csv, steps=250)

    print(f"Saved: {out_summary_csv}")
    print(f"Saved: {out_freq_csv}")
    print(f"Saved: {out_curve_csv}")

    if make_plots:
        plot_coverage_curve(out_curve_csv, out_curve_png)
        plot_loglog_rank_freq(out_freq_csv, out_rank_png)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data / models
    ap.add_argument(
        "--questions_jsonl",
        nargs="+",
        required=False,
        help="One or more JSONL files. Example: --questions_jsonl a.jsonl b.jsonl c.jsonl",
    )
    ap.add_argument("--dpr_q_model", default="sentence-transformers/facebook-dpr-question_encoder-single-nq-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_questions", type=int, default=None)

    # retrieval
    ap.add_argument(
        "--doc_index",
        required=False,
        default="artifacts/wiki-dpr/faiss_wiki_dpr/hnsw_21m/index_psgs_w100_nq_no_index_hnsw_ip_21000000.faiss",
        help="FAISS index used for retrieval",
    )
    ap.add_argument("--truth_index", default=None, help="Optional higher-accuracy index for retrieval")
    ap.add_argument("--use_truth_index", action="store_true", default=False, help="If set and --truth_index provided, use truth_index for retrieval")
    ap.add_argument("--k", type=int, default=20, help="Top-k retrieved per query")

    # outputs
    ap.add_argument("--out_dir", default="runs/chunk_reuse", help="Base output directory")
    ap.add_argument("--make_plots", action="store_true", default=True)

    # misc
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    
    dataset_paths = [
        "data/datasets/qa/nq/nq_train.jsonl",
        "data/datasets/qa/triviaqa/triviaqa_train.jsonl",
        "data/datasets/multiple_choice/openbookqa/openbookqa_train.jsonl",
        "data/datasets/mmlu/all/auxiliary_train.jsonl"
    ]
    args.questions_jsonl = dataset_paths

    # deterministic-ish (only used if you add sampling later)
    np.random.default_rng(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load indices once
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

    # Run each dataset
    for path in args.questions_jsonl:
        run_one_dataset(
            questions_jsonl=path,
            retrieval_index=retrieval_index,
            ntotal=ntotal,
            dpr_q_model=args.dpr_q_model,
            batch_size=args.batch_size,
            device=device,
            max_questions=args.max_questions,
            k=int(args.k),
            out_dir=args.out_dir,
            make_plots=bool(args.make_plots),
        )


if __name__ == "__main__":
    main()
