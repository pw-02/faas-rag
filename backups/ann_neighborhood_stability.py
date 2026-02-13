"""
Drop-in update: sweep M (candidate pool size) and report containment vs M per similarity bin.

If two queries are close in embedding space, do they tend to retrieve documents from the same region of the index?

What changes:
- Add CLI arg: --m_sweep "10,50,100,200,500,1000"
- Modify run_overlap_eval() so it:
  - fetches top max(M_sweep) once per pair
  - computes containment for every M by slicing the ranked list
- Print a compact table per bin.

This is the cleanest way to produce the “you need to cache much more than 10” curve.

Goal: verify the core assumption: nearby queries land in the same ANN region.
You take real queries (best) or synthetic perturbations (ok), and for each pair (q1, q2):

Run a high-recall search for q2 (either exact brute force if N is small enough, or FAISS with very high efSearch / nprobe as a proxy). 
Separately run a normal search for q1 and record a larger set (top-M, like 200/500/1000). 
Now measure: “How many of q2’s true top-k are contained in q1’s top-M?” If for reasonably similar query pairs you see high containment 
(say top-10 is often inside top-500), that tells you the space has local continuity and warm-starting has a chance.

It takes a public QA dataset (your JSONL questions) and embeds them with DPR.
It finds similar questions to each other. For each question q₁, it finds a few nearest neighbor questions q₂.
This gives you pairs: Each pair also has a cosine similarity score like 0.94, 0.97, etc. These are your “nearby queries.”
It groups those pairs by similarity: 0.90–0.93 (kind of similar), 0.93–0.96 (more similar), 0.96–0.98 (very similar), 0.98–1.001 (extremely similar).
Then for each pair:
- Run ANN search for q₁ and collect top M documents (say 500)
- Run a high-accuracy search for q₂ to get its true top K (say 10)
- Then compute: How many of q₂’s true top-10 documents are inside q₁’s top-500? This is called containment.
If containment is high, it means: Nearby queries are landing in the same ANN region, which is good for warm-starting.
i..e, For similar queries, how many of q₂’s true top-10 documents already appear inside q₁’s top-500?

"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import csv
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
    max_length: int,  # unused by ST; kept for compatibility
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


def faiss_topM(index: faiss.Index, q: np.ndarray, M: int) -> np.ndarray:
    _, I = index.search(q.reshape(1, -1), M)
    return I[0]


def exact_topk_ip(doc_vecs: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    sims = doc_vecs @ q.reshape(-1, 1)
    sims = sims.reshape(-1)
    idx = np.argpartition(-sims, k)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx


def containment_rate(true_topk: np.ndarray, candidates: np.ndarray) -> float:
    cand = set(map(int, candidates.tolist()))
    hit = sum(1 for x in true_topk.tolist() if int(x) in cand)
    return hit / max(1, len(true_topk))


def run_overlap_eval_sweep(
    qvecs_raw: np.ndarray,
    pairs_binned: Dict[Tuple[float, float], List[Pair]],
    doc_index: faiss.Index,
    k_true: int,
    m_sweep: List[int],
    max_pairs_per_bin: int,
    doc_vecs_for_truth: Optional[np.ndarray],
    truth_index: Optional[faiss.Index],
    seed: int,
) -> Dict[Tuple[float, float], Dict[int, Dict[str, float]]]:
    """
    Returns: results[bin][M] = {n, mean, p10, p50, p90, sec}
    """
    if (doc_vecs_for_truth is None) == (truth_index is None):
        raise ValueError("Provide exactly one of --doc_vecs_npy (exact truth) OR --truth_index (proxy truth).")

    if not m_sweep:
        raise ValueError("m_sweep must be non-empty.")
    m_sweep = sorted(set(int(x) for x in m_sweep))
    m_max = max(m_sweep)

    rng = np.random.default_rng(seed)
    out: Dict[Tuple[float, float], Dict[int, Dict[str, float]]] = {}

    for b, plist in pairs_binned.items():
        if not plist:
            out[b] = {m: {"n": 0.0, "mean": float("nan")} for m in m_sweep}
            continue

        if len(plist) > max_pairs_per_bin:
            pick = rng.choice(len(plist), size=max_pairs_per_bin, replace=False)
            sample = [plist[int(t)] for t in pick]
        else:
            sample = plist

        # store per-M containments
        conts_by_m: Dict[int, List[float]] = {m: [] for m in m_sweep}
        t0 = time.time()

        desc = f"sweep bin {b[0]:.2f}-{b[1]:.3f} (n={len(sample)})"
        for p in tqdm(sample, desc=desc):
            q1 = qvecs_raw[p.i]
            q2 = qvecs_raw[p.j]

            # fetch once at the max M, slice for smaller Ms
            cand_max = faiss_topM(doc_index, q1, m_max)

            # get "truth" top-k for q2
            if doc_vecs_for_truth is not None:
                true = exact_topk_ip(doc_vecs_for_truth, q2, k_true)
            else:
                _, I2 = truth_index.search(q2.reshape(1, -1), k_true)
                true = I2[0]

            for m in m_sweep:
                conts_by_m[m].append(containment_rate(true, cand_max[:m]))

        dt = time.time() - t0

        out[b] = {}
        for m in m_sweep:
            arr = np.array(conts_by_m[m], dtype=np.float32)
            out[b][m] = {
                "n": float(len(arr)),
                "mean": float(arr.mean()),
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "sec": float(dt),  # same dt for the whole sweep run
            }

    return out


def parse_bins(bins_str: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in bins_str.split(","):
        part = part.strip()
        lo_s, hi_s = part.split("-")
        out.append((float(lo_s), float(hi_s)))
    return out


def parse_int_list(s: str) -> List[int]:
    # "10,50,100" -> [10,50,100]
    return [int(x.strip()) for x in s.split(",") if x.strip()]



def save_results_csv(results, sim_bins, m_sweep, out_path):
    """
    results[bin][M] = {n, mean, p10, p50, p90, sec}
    """
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sim_lo", "sim_hi", "M",
            "n", "mean", "p10", "p50", "p90"
        ])

        for b in sim_bins:
            lo, hi = b
            for m in m_sweep:
                r = results[b][m]
                writer.writerow([
                    lo, hi, m,
                    int(r["n"]),
                    r["mean"],
                    r["p10"],
                    r["p50"],
                    r["p90"],
                ])

def plot_from_csv(csv_path: str, out_png: str):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(7, 5))

    # one curve per similarity bin
    for (lo, hi), g in df.groupby(["sim_lo", "sim_hi"]):
        g = g.sort_values("M")
        label = f"{lo:.2f}–{hi:.2f}"
        plt.plot(g["M"], g["mean"], marker="o", label=label)

    plt.xscale("log")
    plt.xlabel("Candidate pool size M (log scale)")
    plt.ylabel("Containment@K (mean)")
    plt.title("Containment vs Candidate Pool Size")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Cosine similarity")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved plot to {out_png}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_jsonl", default="data/datasets/qa/nq/nq_dev.jsonl")
    ap.add_argument("--doc_index", default="artifacts/index_psgs_w100_nq_no_index_ivf_ip_1000000.faiss")
    ap.add_argument("--truth_index", default="artifacts/index_psgs_w100_nq_no_index_flat_ip_1000000.faiss")
    ap.add_argument("--doc_vecs_npy", default=None)
    ap.add_argument("--dpr_q_model", default="sentence-transformers/facebook-dpr-question_encoder-single-nq-base")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_questions", type=int, default=50000)
    ap.add_argument("--neighbors_per_query", type=int, default=3)
    ap.add_argument("--min_sim", type=float, default=0.90)
    ap.add_argument("--max_pairs", type=int, default=50000)
    ap.add_argument("--pair_metric", choices=["ip", "cosine"], default="cosine")
    ap.add_argument("--bins", default="0.90-0.93,0.93-0.96,0.96-0.98,0.98-1.001")
    ap.add_argument("--k_true", type=int, default=10)
    ap.add_argument("--m_sweep", default="10,50,100,200,500,1000",
                    help="Comma-separated Ms to evaluate, e.g. '10,50,100,200,500,1000'")
    ap.add_argument("--max_pairs_per_bin", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_csv", default="containment_sweep.csv",
                help="Path to write CSV results")
    ap.add_argument("--make_plot", action="store_true", default=True,
                help="If set, generate containment-vs-M figure")
    ap.add_argument("--plot_png", default="containment_vs_M.png",
                help="Output PNG for plot")

    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sim_bins = parse_bins(args.bins)
    m_sweep = parse_int_list(args.m_sweep)

    # Load indices
    doc_index = faiss.read_index(args.doc_index)

    doc_vecs_for_truth: Optional[np.ndarray] = None
    truth_index: Optional[faiss.Index] = None
    if args.doc_vecs_npy and args.truth_index:
        raise SystemExit("Choose only one: --doc_vecs_npy OR --truth_index.")
    if (not args.doc_vecs_npy) and (not args.truth_index):
        raise SystemExit("Provide one of: --doc_vecs_npy (exact truth) OR --truth_index (proxy truth).")

    if args.doc_vecs_npy:
        doc_vecs_for_truth = np.load(args.doc_vecs_npy).astype(np.float32)
    else:
        truth_index = faiss.read_index(args.truth_index)

    # Load questions
    questions = load_questions_from_jsonl(args.questions_jsonl, max_q=args.max_questions)
    if len(questions) < 2:
        raise SystemExit("Need at least 2 questions.")
    print(f"Loaded {len(questions)} questions from {args.questions_jsonl}")

    # Embed queries (RAW)
    t0 = time.time()
    qvecs_raw = embed_questions_dpr_raw(
        questions=questions,
        model_name=args.dpr_q_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    print(f"Embedded queries (raw/IP): shape={qvecs_raw.shape} in {time.time()-t0:.1f}s")

    # Pairing space
    if args.pair_metric == "ip":
        qvecs_pair = qvecs_raw
        print("Pairing metric: IP on raw query vectors.")
    else:
        qvecs_pair = l2_normalize(qvecs_raw)
        print("Pairing metric: cosine on normalized query vectors.")

    # Build query NN index + pairs
    q_index = build_query_index_ip(qvecs_pair)
    pairs = make_query_pairs_from_nn(
        qvecs_for_pairing=qvecs_pair,
        q_index=q_index,
        neighbors_per_query=args.neighbors_per_query,
        min_sim=args.min_sim,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"Created {len(pairs)} query pairs with sim >= {args.min_sim} under pair_metric={args.pair_metric}")

    # Bin
    pairs_binned = bin_pairs(pairs, sim_bins)
    for b in sim_bins:
        print(f"Bin {b}: {len(pairs_binned[b])} pairs")

    # Sweep eval
    results = run_overlap_eval_sweep(
        qvecs_raw=qvecs_raw,
        pairs_binned=pairs_binned,
        doc_index=doc_index,
        k_true=args.k_true,
        m_sweep=m_sweep,
        max_pairs_per_bin=args.max_pairs_per_bin,
        doc_vecs_for_truth=doc_vecs_for_truth,
        truth_index=truth_index,
        seed=args.seed,
    )

    # Print
    print("\n=== Containment@K vs M sweep (IP retrieval; pairing by {}) ===".format(args.pair_metric))
    print(f"K_true={args.k_true}  M_sweep={m_sweep}  max_pairs_per_bin={args.max_pairs_per_bin}")
    for b in sim_bins:
        print(f"\nSimilarity bin {b}:")
        for m in m_sweep:
            r = results[b][m]
            n = int(r.get("n", 0))
            if n == 0:
                print(f"  M={m:4d}: n=0")
            else:
                print(
                    f"  M={m:4d}: n={n:4d} mean={r['mean']:.3f} "
                    f"p10={r['p10']:.3f} p50={r['p50']:.3f} p90={r['p90']:.3f}"
                )

    # Save CSV
    if args.out_csv:
        save_results_csv(results, sim_bins, m_sweep, args.out_csv)
        print(f"\nSaved results to {args.out_csv}")
    
    if args.make_plot:
        plot_from_csv(args.out_csv, args.plot_png)


if __name__ == "__main__":
    main()
