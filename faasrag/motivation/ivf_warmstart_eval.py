#!/usr/bin/env python3
"""
ivf_warmstart_eval.py

Purpose
-------
Show a *speed/compute benefit* from "caching the search region" for IVF indexes.

We do this without modifying FAISS internals by using IVF's Python-exposed
`search_preassigned` (if your FAISS build exposes it).

Workflow
--------
1) Load questions from JSONL (expects field "question")
2) Embed with DPR question encoder (SentenceTransformers DPR wrapper)
3) Build "nearby query pairs" by cosine similarity (on normalized query vectors)
4) Bin pairs by cosine similarity
5) Evaluate retrieval for q2 with:
   - Baseline: IVF.search(q2) with nprobe = P
   - Warm-start: IVF.search_preassigned(q2) forced to probe the SAME coarse lists
     that q1 would probe under nprobe = P (i.e., "reuse q1 region" as a warm start)

Truth
-----
Use a Flat-IP (exact) index for truth top-K on docs (ground truth retrieval).

Inputs you need
---------------
- --doc_index_ivf : IVF index (IVFFlat or IVFPQ) built for INNER PRODUCT
- --truth_index   : Flat-IP exact index (same vectors)
- --questions_jsonl : dataset of questions

Outputs
-------
- CSV with recall@K and latency (ms) for baseline vs warm-start, per similarity bin.
- Optional plot (PNG) if --make_plot

Notes
-----
- This is a *region reuse* experiment (lists-to-probe reuse), not final "path caching"
  inside HNSW. It's the cleanest IVF analogue and produces a compelling figure.

  If they land in the same region, can we actually exploit that to reduce ANN work while keeping recall?

  Your first experiment only showed geometry: it showed that when two questions are close in embedding space, the documents relevant to the second question usually appear somewhere inside the neighborhood retrieved for the first. That told you there is locality in the system. But it didn’t prove that this locality is useful for speeding anything up.

This second experiment is about exploiting that locality.

Here, you take pairs of similar queries (q1, q2). For q2, you compare two ways of searching the IVF index.

In the baseline case, q2 is treated like a brand-new query. IVF compares q2 against all coarse centroids, chooses which inverted lists to probe, and then scans those lists. This is the normal cold-start behavior. A lot of time is spent just figuring out which region of the index to look in.

In the warm-start case, you don’t let IVF rediscover the region. Instead, you reuse the inverted lists that q1 already selected. In other words, you tell IVF: “start searching in the same coarse region that the previous similar query used.” Then q2 does the same fine-grained scoring and ranking as usual, but it skips the expensive global discovery step.

Nothing else changes. You’re not changing k. You’re not changing ranking. You’re not dropping candidates. You’re only changing how the search is initialized.

For each query pair, you measure how long baseline search takes, how long warm-start search takes, and whether both recover the true top documents (using your Flat index as ground truth). You then average these numbers within similarity bins.

If warm-start gives similar recall but lower latency, that’s the key result. It shows that cached search regions can reduce ANN work without hurting retrieval quality.

This directly connects back to your earlier containment experiment. That experiment showed that q2’s answers usually live inside q1’s neighborhood, but not necessarily inside q1’s top-10. That means simple answer caching is unreliable, but region reuse is promising. This IVF experiment demonstrates that promise in practice by showing that starting from the previous region avoids wasted exploration.

So conceptually, the story is: similar queries land in the same ANN regions; cold search wastes time rediscovering those regions; warm-start skips that rediscovery; therefore retrieval gets faster while staying accurate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------
# Data + Embeddings
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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@torch.no_grad()
def embed_questions_dpr_raw_st(
    questions: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    DPR question embeddings WITHOUT normalization (raw vectors),
    using SentenceTransformers wrapper.

    IMPORTANT: Use ST-wrapped checkpoint:
      sentence-transformers/facebook-dpr-question_encoder-single-nq-base
    """
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    chunks: List[np.ndarray] = []
    for s in range(0, len(questions), batch_size):
        batch = questions[s : s + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        chunks.append(emb)
    return np.vstack(chunks).astype(np.float32, copy=False)


# -----------------------------
# Query pairing
# -----------------------------
@dataclass
class Pair:
    i: int
    j: int
    sim: float  # cosine similarity (we use normalized query vectors for pairing)


def build_query_index_cosine(qvecs_norm: np.ndarray) -> faiss.Index:
    """Cosine NN over normalized vectors via inner product."""
    d = qvecs_norm.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(qvecs_norm)
    return idx


def make_query_pairs_from_nn(
    qvecs_norm: np.ndarray,
    q_index: faiss.Index,
    neighbors_per_query: int,
    min_sim: float,
    max_pairs: int,
    seed: int,
) -> List[Pair]:
    rng = np.random.default_rng(seed)
    n = qvecs_norm.shape[0]
    order = np.arange(n)
    rng.shuffle(order)

    pairs: List[Pair] = []
    k = neighbors_per_query + 1  # include self
    for i in order:
        q = qvecs_norm[i : i + 1]
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


def parse_bins(bins_str: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in bins_str.split(","):
        part = part.strip()
        lo_s, hi_s = part.split("-")
        out.append((float(lo_s), float(hi_s)))
    return out


def bin_pairs(pairs: List[Pair], bins: List[Tuple[float, float]]) -> Dict[Tuple[float, float], List[Pair]]:
    out: Dict[Tuple[float, float], List[Pair]] = {b: [] for b in bins}
    for p in pairs:
        for lo, hi in bins:
            if lo <= p.sim < hi:
                out[(lo, hi)].append(p)
                break
    return out


# -----------------------------
# IVF warm-start mechanics
# -----------------------------
def require_ivf(index: faiss.Index) -> faiss.IndexIVF:
    ivf = faiss.downcast_index(index)
    if not isinstance(ivf, faiss.IndexIVF):
        raise ValueError("Provided doc index is not an IVF index (IndexIVF / IndexIVFFlat / IndexIVFPQ).")
    return ivf


def get_ivf_lists_for_query(ivf: faiss.IndexIVF, q: np.ndarray, nprobe: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the coarse list IDs q would probe under quantizer.search, plus their scores.

    For IP IVF, quantizer.search returns (scores, list_ids) where scores are inner products
    between q and coarse centroids (used internally for list ordering).
    """
    scores, ids = ivf.quantizer.search(q.reshape(1, -1), nprobe)
    return ids[0].astype(np.int64), scores[0].astype(np.float32)


def search_preassigned(ivf: faiss.IndexIVF, q: np.ndarray, k: int,
                       list_ids: np.ndarray, list_scores: np.ndarray) -> np.ndarray:
    """
    Force IVF to search ONLY the provided lists, using Python-exposed search_preassigned.
    Returns doc IDs (k,).

    If your FAISS build doesn't expose this, we raise with a clear error.
    """
    if not hasattr(ivf, "search_preassigned"):
        raise RuntimeError(
            "FAISS Python bindings do not expose IndexIVF.search_preassigned in your build.\n"
            "Options:\n"
            "  (1) install a different FAISS build that includes search_preassigned,\n"
            "  (2) do a weaker experiment: 'warm lists first then add normal lists' via two-phase rerank,\n"
            "  (3) add a small C++ hook.\n"
        )

    assign = list_ids.reshape(1, -1).astype(np.int64, copy=False)
    coarse = list_scores.reshape(1, -1).astype(np.float32, copy=False)

    # Some builds accept output arrays, some return (D, I)
    D = np.empty((1, k), dtype=np.float32)
    I = np.empty((1, k), dtype=np.int64)
    try:
        ivf.search_preassigned(q.reshape(1, -1), k, assign, coarse, D, I)
        return I[0]
    except TypeError:
        D2, I2 = ivf.search_preassigned(q.reshape(1, -1), k, assign, coarse)
        return I2[0]


def recall_at_k(pred_ids: np.ndarray, true_ids: np.ndarray) -> float:
    s = set(map(int, pred_ids.tolist()))
    hit = sum(1 for x in true_ids.tolist() if int(x) in s)
    return hit / max(1, len(true_ids))


# -----------------------------
# Evaluation
# -----------------------------
def eval_ivf_warmstart(
    ivf: faiss.IndexIVF,
    truth: faiss.Index,
    qvecs_raw: np.ndarray,
    pairs_binned: Dict[Tuple[float, float], List[Pair]],
    nprobe: int,
    k_true: int,
    k_eval: int,
    max_pairs_per_bin: int,
    seed: int,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """
    For each similarity bin:
      baseline: ivf.search(q2, k_eval) with ivf.nprobe=nprobe
      warm:     ivf.search_preassigned(q2, k_eval, lists_from_q1)

    Returns a dict per bin with:
      n, baseline_recall, warm_recall, baseline_ms, warm_ms
    """
    rng = np.random.default_rng(seed)

    # baseline nprobe
    ivf.nprobe = int(nprobe)

    results: Dict[Tuple[float, float], Dict[str, float]] = {}

    for b, plist in pairs_binned.items():
        if not plist:
            results[b] = {"n": 0.0}
            continue

        if len(plist) > max_pairs_per_bin:
            pick = rng.choice(len(plist), size=max_pairs_per_bin, replace=False)
            sample = [plist[int(t)] for t in pick]
        else:
            sample = plist

        base_rec, warm_rec = [], []
        base_t, warm_t = [], []

        desc = f"eval bin {b[0]:.2f}-{b[1]:.3f} (n={len(sample)})"
        for p in tqdm(sample, desc=desc):
            q1 = qvecs_raw[p.i]
            q2 = qvecs_raw[p.j]

            # truth for q2
            _, trueI = truth.search(q2.reshape(1, -1), k_true)
            true_ids = trueI[0]

            # baseline
            t0 = time.perf_counter()
            _, baseI = ivf.search(q2.reshape(1, -1), k_eval)
            base_dt = (time.perf_counter() - t0) * 1000.0
            base_ids = baseI[0]
            base_t.append(base_dt)
            base_rec.append(recall_at_k(base_ids, true_ids))

            # warm-start lists = region from q1
            list_ids, list_scores = get_ivf_lists_for_query(ivf, q1, nprobe)
            t1 = time.perf_counter()
            warm_ids = search_preassigned(ivf, q2, k_eval, list_ids, list_scores)
            warm_dt = (time.perf_counter() - t1) * 1000.0
            warm_t.append(warm_dt)
            warm_rec.append(recall_at_k(warm_ids, true_ids))

        results[b] = {
            "n": float(len(sample)),
            "baseline_recall": float(np.mean(base_rec)),
            "warm_recall": float(np.mean(warm_rec)),
            "baseline_ms": float(np.mean(base_t)),
            "warm_ms": float(np.mean(warm_t)),
        }

    return results


def save_csv(results: Dict[Tuple[float, float], Dict[str, float]],
             out_csv: str,
             index_label: str,
             nprobe: int,
             k_true: int,
             k_eval: int):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    header = [
        "index", "nprobe", "k_true", "k_eval",
        "sim_lo", "sim_hi", "n",
        "baseline_recall", "warm_recall",
        "baseline_ms", "warm_ms",
        "speedup_x",
    ]
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for (lo, hi), r in results.items():
            n = int(r.get("n", 0))
            if n == 0:
                continue
            bms = r["baseline_ms"]
            wms = r["warm_ms"]
            speedup = (bms / wms) if (wms > 0) else float("nan")
            w.writerow([
                index_label, nprobe, k_true, k_eval,
                lo, hi, n,
                r["baseline_recall"], r["warm_recall"],
                bms, wms,
                speedup,
            ])


def maybe_plot(csv_path: str, out_png: str):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    # Plot speedup per bin (mean over rows if multiple nprobe/index)
    plt.figure(figsize=(8, 5))
    for (idx, nprobe), g in df.groupby(["index", "nprobe"]):
        # sort bins by sim_lo
        g = g.sort_values(["sim_lo", "sim_hi"])
        x = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(g["sim_lo"], g["sim_hi"])]
        plt.plot(x, g["speedup_x"], marker="o", label=f"{idx} nprobe={nprobe}")

    plt.xlabel("Cosine similarity bin (q1,q2)")
    plt.ylabel("Speedup (baseline_ms / warm_ms)")
    plt.title("IVF warm-start speedup by similarity bin")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Saved plot to {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--questions_jsonl", default="data/datasets/qa/nq/nq_dev.jsonl", help="JSONL with field 'question'")
    ap.add_argument("--max_questions", type=int, default=50000)

    ap.add_argument("--dpr_q_model",
                    default="sentence-transformers/facebook-dpr-question_encoder-single-nq-base",
                    help="ST DPR question encoder checkpoint")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--neighbors_per_query", type=int, default=3)
    ap.add_argument("--min_sim", type=float, default=0.90)
    ap.add_argument("--max_pairs", type=int, default=50000)
    ap.add_argument("--bins", default="0.90-0.93,0.93-0.96,0.96-0.98,0.98-1.001")

    ap.add_argument("--doc_index_ivf", default="artifacts/index_psgs_w100_nq_no_index_ivf_ip_1000000.faiss", help="IVF index (IVFFlat or IVFPQ), IP metric")
    ap.add_argument("--truth_index", default="artifacts/index_psgs_w100_nq_no_index_flat_ip_1000000.faiss", help="Flat-IP exact index for truth")

    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--k_true", type=int, default=10)
    ap.add_argument("--k_eval", type=int, default=10)
    ap.add_argument("--max_pairs_per_bin", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--index_label", default=None, help="Label for CSV rows (e.g., IVFFlat or IVFPQ)")
    ap.add_argument("--out_csv", default="ivf_warmstart_results.csv")

    ap.add_argument("--make_plot", action="store_true")
    ap.add_argument("--plot_png", default="ivf_warmstart_speedup.png")

    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sim_bins = parse_bins(args.bins)

    # Load questions + embed
    questions = load_questions_from_jsonl(args.questions_jsonl, max_q=args.max_questions)
    if len(questions) < 2:
        raise SystemExit("Need at least 2 questions.")
    print(f"Loaded {len(questions)} questions")

    t0 = time.time()
    qraw = embed_questions_dpr_raw_st(
        questions=questions,
        model_name=args.dpr_q_model,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Embedded queries: {qraw.shape} in {time.time()-t0:.1f}s on {device}")

    # Build query pairs by cosine similarity
    qnorm = l2_normalize(qraw)
    q_index = build_query_index_cosine(qnorm)
    pairs = make_query_pairs_from_nn(
        qvecs_norm=qnorm,
        q_index=q_index,
        neighbors_per_query=args.neighbors_per_query,
        min_sim=args.min_sim,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"Created {len(pairs)} query pairs with cosine >= {args.min_sim}")

    pairs_binned = bin_pairs(pairs, sim_bins)
    for b in sim_bins:
        print(f"Bin {b}: {len(pairs_binned[b])} pairs")

    # Load IVF + truth
    ivf_raw = faiss.read_index(args.doc_index_ivf)
    ivf = require_ivf(ivf_raw)
    truth = faiss.read_index(args.truth_index)

    if args.index_label:
        label = args.index_label
    else:
        # heuristic label
        label = "IVF"
        tname = type(ivf).__name__
        if "PQ" in tname:
            label = "IVFPQ"
        elif "Flat" in tname:
            label = "IVFFlat"

    # Run eval
    print(f"\nEvaluating IVF warm-start: index={label}, nprobe={args.nprobe}, k_eval={args.k_eval}, k_true={args.k_true}")
    res = eval_ivf_warmstart(
        ivf=ivf,
        truth=truth,
        qvecs_raw=qraw,
        pairs_binned=pairs_binned,
        nprobe=args.nprobe,
        k_true=args.k_true,
        k_eval=args.k_eval,
        max_pairs_per_bin=args.max_pairs_per_bin,
        seed=args.seed,
    )

    # Print summary
    print("\n=== Results (mean over sampled pairs) ===")
    for b in sim_bins:
        r = res.get(b, {"n": 0})
        n = int(r.get("n", 0))
        if n == 0:
            print(f"sim {b}: n=0")
            continue
        speedup = r["baseline_ms"] / r["warm_ms"] if r["warm_ms"] > 0 else float("nan")
        print(
            f"sim {b}: n={n} "
            f"base_recall={r['baseline_recall']:.3f} warm_recall={r['warm_recall']:.3f} "
            f"base_ms={r['baseline_ms']:.3f} warm_ms={r['warm_ms']:.3f} "
            f"speedup={speedup:.2f}x"
        )

    # Save CSV (append)
    save_csv(
        results=res,
        out_csv=args.out_csv,
        index_label=label,
        nprobe=args.nprobe,
        k_true=args.k_true,
        k_eval=args.k_eval,
    )
    print(f"\nAppended CSV rows to {args.out_csv}")

    if args.make_plot:
        maybe_plot(args.out_csv, args.plot_png)


if __name__ == "__main__":
    main()
