from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@dataclass
class Pair:
    i: int
    j: int
    sim: float


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
def embed_questions_dpr_raw(
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


def build_query_index_ip(qvecs: np.ndarray) -> faiss.Index:
    d = qvecs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(qvecs)
    return idx


def make_query_pairs_from_nn(
    qvecs_for_pairing: np.ndarray,
    q_index: faiss.Index,
    neighbors_per_query: int,
    min_sim: float,
    max_pairs: int,
    seed: int,
) -> List[Pair]:
    rng = np.random.default_rng(seed)
    n = qvecs_for_pairing.shape[0]
    order = np.arange(n)
    rng.shuffle(order)

    pairs: List[Pair] = []
    k = neighbors_per_query + 1  # include self

    for i in order:
        q = qvecs_for_pairing[i : i + 1]
        D, I = q_index.search(q, k)
        for sim, j in zip(D[0], I[0]):
            if j < 0 or j == i:
                continue
            simf = float(sim)
            if simf >= min_sim:
                pairs.append(Pair(i=int(i), j=int(j), sim=simf))
        if len(pairs) >= max_pairs:
            break

    return pairs


def bin_pairs(pairs: List[Pair], bins: List[Tuple[float, float]]) -> Dict[Tuple[float, float], List[Pair]]:
    out: Dict[Tuple[float, float], List[Pair]] = {b: [] for b in bins}
    for p in pairs:
        placed = False
        for lo, hi in bins:
            if lo <= p.sim < hi:
                out[(lo, hi)].append(p)
                placed = True
                break
        if not placed and p.sim >= bins[-1][1]:
            out[bins[-1]].append(p)
    return out


def faiss_topk(index: faiss.Index, q: np.ndarray, k: int) -> np.ndarray:
    _, I = index.search(q.reshape(1, -1), k)
    return I[0]


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(map(int, a.tolist()))
    sb = set(map(int, b.tolist()))
    inter = len(sa & sb)
    k = max(1, len(a))
    return inter / k


def jaccard_at_k(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(map(int, a.tolist()))
    sb = set(map(int, b.tolist()))
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(1, union)


def rbo_at_k(a: np.ndarray, b: np.ndarray, p: float = 0.9) -> float:
    k = min(len(a), len(b))
    if k <= 0:
        return float("nan")

    seen_a = set()
    seen_b = set()
    summation = 0.0

    for d in range(1, k + 1):
        seen_a.add(int(a[d - 1]))
        seen_b.add(int(b[d - 1]))
        overlap = len(seen_a & seen_b)
        summation += (overlap / d) * (p ** (d - 1))

    return (1.0 - p) * summation


def parse_bins(bins_str: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in bins_str.split(","):
        part = part.strip()
        lo_s, hi_s = part.split("-")
        out.append((float(lo_s), float(hi_s)))
    return out


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def run_semcache_eval(
    qvecs_raw: np.ndarray,
    pairs_binned: Dict[Tuple[float, float], List[Pair]],
    doc_index: faiss.Index,
    k_list: List[int],
    max_pairs_per_bin: int,
    seed: int,
    rbo_p: float,
    truth_index: Optional[faiss.Index] = None,
    truth_for: str = "both",
) -> Dict[Tuple[float, float], Dict[int, Dict[str, Dict[str, float]]]]:
    rng = np.random.default_rng(seed)
    k_list = sorted(set(int(k) for k in k_list))
    k_max = max(k_list)

    out: Dict[Tuple[float, float], Dict[int, Dict[str, Dict[str, float]]]] = {}

    for b, plist in pairs_binned.items():
        if not plist:
            out[b] = {
                k: {
                    m: {"n": 0.0, "mean": float("nan"), "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
                    for m in ["overlap", "mismatch", "jaccard", "rbo"]
                }
                for k in k_list
            }
            continue

        if len(plist) > max_pairs_per_bin:
            pick = rng.choice(len(plist), size=max_pairs_per_bin, replace=False)
            sample = [plist[int(t)] for t in pick]
        else:
            sample = plist

        metrics_by_k = {k: {m: [] for m in ["overlap", "mismatch", "jaccard", "rbo"]} for k in k_list}

        desc = f"semcache bin {b[0]:.2f}-{b[1]:.3f} (n={len(sample)})"
        t0 = time.time()
        for p in tqdm(sample, desc=desc):
            q1 = qvecs_raw[p.i]
            q2 = qvecs_raw[p.j]

            if truth_index is None or truth_for == "none":
                r1_max = faiss_topk(doc_index, q1, k_max)
                r2_max = faiss_topk(doc_index, q2, k_max)
            elif truth_for == "q1":
                r1_max = faiss_topk(truth_index, q1, k_max)
                r2_max = faiss_topk(doc_index, q2, k_max)
            elif truth_for == "both":
                r1_max = faiss_topk(truth_index, q1, k_max)
                r2_max = faiss_topk(truth_index, q2, k_max)
            else:
                raise ValueError("--truth_for must be one of: none, q1, both")

            for k in k_list:
                r1 = r1_max[:k]
                r2 = r2_max[:k]
                ov = overlap_at_k(r1, r2)
                jc = jaccard_at_k(r1, r2)
                rb = rbo_at_k(r1, r2, p=rbo_p)

                metrics_by_k[k]["overlap"].append(ov)
                metrics_by_k[k]["mismatch"].append(1.0 - ov)
                metrics_by_k[k]["jaccard"].append(jc)
                metrics_by_k[k]["rbo"].append(rb)

        dt = time.time() - t0

        out[b] = {}
        for k in k_list:
            out[b][k] = {}
            for m in ["overlap", "mismatch", "jaccard", "rbo"]:
                arr = np.array(metrics_by_k[k][m], dtype=np.float32)
                summ = summarize(arr)
                summ["sec"] = float(dt)
                out[b][k][m] = summ

    return out


def save_semcache_csv(results, sim_bins, k_list, out_path: str):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_lo", "sim_hi", "k", "metric", "n", "mean", "p10", "p50", "p90"])
        for b in sim_bins:
            lo, hi = b
            for k in k_list:
                for metric, summ in results[b][k].items():
                    w.writerow([lo, hi, k, metric, int(summ["n"]), summ["mean"], summ["p10"], summ["p50"], summ["p90"]])


def compute_bin_stats(
    pairs_binned: Dict[Tuple[float, float], List[Pair]],
    num_queries_total: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for (lo, hi), plist in pairs_binned.items():
        qs = set()
        for p in plist:
            qs.add(p.i)
            qs.add(p.j)
        rows.append(
            {
                "sim_lo": float(lo),
                "sim_hi": float(hi),
                "pair_count": int(len(plist)),
                "unique_queries_in_bin": int(len(qs)),
                "unique_query_frac": float(len(qs) / max(1, num_queries_total)),
            }
        )
    rows.sort(key=lambda r: (r["sim_lo"], r["sim_hi"]))
    return rows


def save_bin_stats_csv(bin_rows: List[Dict[str, float]], out_path: str):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_lo", "sim_hi", "pair_count", "unique_queries_in_bin", "unique_query_frac"])
        for r in bin_rows:
            w.writerow([r["sim_lo"], r["sim_hi"], r["pair_count"], r["unique_queries_in_bin"], r["unique_query_frac"]])


def plot_metric_from_csv(csv_path: str, out_png: str, metric: str):
    df = pd.read_csv(csv_path)
    df = df[df["metric"] == metric].copy()
    if df.empty:
        raise SystemExit(f"No rows for metric={metric} in {csv_path}")

    plt.figure(figsize=(8, 5))
    for (lo, hi), g in df.groupby(["sim_lo", "sim_hi"]):
        g = g.sort_values("k")
        label = f"{lo:.2f}–{hi:.2f}"
        plt.plot(g["k"], g["mean"], marker="o", label=label)

    plt.xlabel("k (top-k retrieved)")
    plt.ylabel(f"{metric} (mean)")
    plt.title(f"{metric} vs k by similarity bin")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Cosine similarity bin")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")


def plot_bin_stats_csv(bin_csv_path: str, out_png: str):
    df = pd.read_csv(bin_csv_path).sort_values(["sim_lo", "sim_hi"]).copy()
    if df.empty:
        raise SystemExit(f"No rows in {bin_csv_path}")

    plt.figure(figsize=(8, 4))
    labels = [f"{r.sim_lo:.2f}–{r.sim_hi:.3f}" for r in df.itertuples(index=False)]
    plt.bar(labels, df["unique_query_frac"].values)
    plt.ylabel("Unique query fraction (proxy hit rate)")
    plt.xlabel("Cosine similarity bin")
    plt.title("Approximate semantic cache hit rate by similarity bin")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved bin-stats plot to {out_png}")


def dataset_name_from_path(path: str) -> str:
    """
    Make a safe directory name from the dataset file.
    Examples:
      /a/b/nq_dev.jsonl -> nq_dev
      triviaqa.jsonl    -> triviaqa
    """
    base = os.path.basename(path)
    name = re.sub(r"\.jsonl$", "", base, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "dataset"


def run_one_dataset(args, questions_jsonl: str, doc_index: faiss.Index, truth_index: Optional[faiss.Index]) -> None:
    dname = dataset_name_from_path(questions_jsonl)
    out_dir = os.path.join(args.out_dir, dname)
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "semcache_mismatch.csv")
    out_bins_csv = os.path.join(out_dir, "semcache_bins.csv")
    plot_png = os.path.join(out_dir, "metric_vs_k.png")
    bins_plot_png = os.path.join(out_dir, "bin_hit_rate.png")

    print("\n" + "=" * 90)
    print(f"DATASET: {dname}")
    print(f"INPUT : {questions_jsonl}")
    print(f"OUTDIR: {out_dir}")
    print("=" * 90)

    # Load questions
    questions = load_questions_from_jsonl(questions_jsonl, max_q=args.max_questions)
    if len(questions) < 2:
        print(f"Skipping {questions_jsonl}: need at least 2 questions.")
        return
    print(f"Loaded {len(questions)} questions")

    # Embed queries
    t0 = time.time()
    qvecs_raw = embed_questions_dpr_raw(
        questions=questions,
        model_name=args.dpr_q_model,
        batch_size=args.batch_size,
        device=args._resolved_device,
    )
    print(f"Embedded queries: shape={qvecs_raw.shape} in {time.time()-t0:.1f}s")

    # Pairing space
    if args.pair_metric == "ip":
        qvecs_pair = qvecs_raw
        print("Pairing metric: IP on raw query vectors.")
    else:
        qvecs_pair = l2_normalize(qvecs_raw)
        print("Pairing metric: cosine on normalized query vectors.")

    # Build query NN index + pairs
    sim_bins = parse_bins(args.bins)
    k_list = parse_int_list(args.k_list)

    q_index = build_query_index_ip(qvecs_pair)
    pairs = make_query_pairs_from_nn(
        qvecs_for_pairing=qvecs_pair,
        q_index=q_index,
        neighbors_per_query=args.neighbors_per_query,
        min_sim=args.min_sim,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"Created {len(pairs)} query pairs with sim >= {args.min_sim}")

    pairs_binned = bin_pairs(pairs, sim_bins)
    for b in sim_bins:
        print(f"Bin {b}: {len(pairs_binned[b])} pairs")

    # Bin stats (proxy cache hit rate)
    bin_rows = compute_bin_stats(pairs_binned, num_queries_total=qvecs_raw.shape[0])
    if args.out_bins_csv:
        save_bin_stats_csv(bin_rows, out_bins_csv)
        print(f"Saved bin stats to {out_bins_csv}")
    if args.make_bins_plot:
        plot_bin_stats_csv(out_bins_csv, bins_plot_png)

    # Eval
    results = run_semcache_eval(
        qvecs_raw=qvecs_raw,
        pairs_binned=pairs_binned,
        doc_index=doc_index,
        k_list=k_list,
        max_pairs_per_bin=args.max_pairs_per_bin,
        seed=args.seed,
        rbo_p=args.rbo_p,
        truth_index=truth_index,
        truth_for=args.truth_for,
    )

    # Save results + plot
    if args.out_csv:
        save_semcache_csv(results, sim_bins, k_list, out_csv)
        print(f"Saved results to {out_csv}")

    if args.make_plot:
        plot_metric_from_csv(out_csv, plot_png, args.plot_metric)

    # Console summary (optional)
    print("\nSummary (mismatch mean / median by bin):")
    for b in sim_bins:
        lo, hi = b
        line = [f"{lo:.2f}-{hi:.3f}"]
        for k in k_list:
            mm = results[b][k]["mismatch"]
            if int(mm.get("n", 0)) == 0:
                line.append(f"k={k}:n=0")
            else:
                line.append(f"k={k}:mean={mm['mean']:.3f},p50={mm['p50']:.3f}")
        print("  " + " | ".join(line))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data / models
    ap.add_argument(
        "--questions_jsonl",
        nargs="+",
        help="One or more JSONL files. Example: --questions_jsonl a.jsonl b.jsonl c.jsonl",
    )
    ap.add_argument("--dpr_q_model", default="sentence-transformers/facebook-dpr-question_encoder-single-nq-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_questions", type=int, default=50000)

    # indices
    ap.add_argument(
        "--doc_index",
        default="artifacts/wiki-dpr/faiss_wiki_dpr/hnsw_21m/index_psgs_w100_nq_no_index_hnsw_ip_21000000.faiss",
        help="Main ANN index used for retrieval (simulates production index)",
    )
    ap.add_argument("--truth_index", default=None, help="Optional: higher-accuracy index (FlatIP / high-efSearch) for ground truth")
    ap.add_argument("--truth_for", choices=["none", "q1", "both"], default="none", help="If --truth_index is given, which side uses it: none|q1|both")

    # pairing
    ap.add_argument("--neighbors_per_query", type=int, default=3)
    ap.add_argument("--min_sim", type=float, default=0.90)
    ap.add_argument("--max_pairs", type=int, default=100_000)
    ap.add_argument("--pair_metric", choices=["ip", "cosine"], default="cosine")
    ap.add_argument("--bins", default="0.90-0.93,0.93-0.96,0.96-0.98,0.98-1.001")

    # eval
    ap.add_argument("--k_list", default="1,5,10,20,40,60", help="Comma-separated k values for overlap/mismatch/etc")
    ap.add_argument("--max_pairs_per_bin", type=int, default=500)
    ap.add_argument("--rbo_p", type=float, default=0.9)

    # output
    ap.add_argument("--out_dir", default="runs/semcache_eval", help="Base directory to write per-dataset results")
    ap.add_argument("--out_csv", action="store_true", default=True, help="Write semcache_mismatch.csv per dataset")
    ap.add_argument("--out_bins_csv", action="store_true", default=True, help="Write semcache_bins.csv per dataset")
    ap.add_argument("--make_plot", action="store_true", default=True)
    ap.add_argument("--plot_metric", choices=["mismatch", "overlap", "jaccard", "rbo"], default="mismatch")
    ap.add_argument("--make_bins_plot", action="store_true", default=True)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    args._resolved_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load indices ONCE (shared across datasets)
    doc_index = faiss.read_index(args.doc_index)
    truth_index = faiss.read_index(args.truth_index) if args.truth_index else None

    dataset_paths = [
        "data/datasets/qa/nq/nq_train.jsonl",
        "data/datasets/qa/triviaqa/triviaqa_train.jsonl",
        "data/datasets/multiple_choice/openbookqa/openbookqa_train.jsonl",
        "data/datasets/mmlu/all/auxiliary_train.jsonl"
    ]
    args.questions_jsonl = dataset_paths

    # Run each dataset
    for path in args.questions_jsonl:
        run_one_dataset(args, path, doc_index=doc_index, truth_index=truth_index)


if __name__ == "__main__":
    main()
