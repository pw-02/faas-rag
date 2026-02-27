#!/usr/bin/env python3
"""
Microbenchmark: brute-force (dense matrix dot-products) vs FAISS ANN.

Measures per-query latency (p50/p95 across repetitions) while sweeping N vectors.
Baselines:
- Brute-force exact search over a contiguous matrix (IP/L2; cosine via normalization).
- FAISS HNSW (IndexHNSWFlat) (optional)
- FAISS IVF-Flat (IndexIVFFlat) (optional)
- FAISS IVF-PQ (IndexIVFPQ) (optional)

Optional: compute Recall@K vs exact brute force (can be slow for large N).

CPU-only brute force. FAISS GPU indices are not used here (but you can install faiss-gpu and adapt).

Examples:
  python bench_vec_cache_full.py --metric cosine --dtype float16 --batch 16 \
    --use_hnsw --use_ivfflat --ivf_nlist 4096 --ivf_nprobe 16 \
    --min_n 50000 --max_n 800000 --steps 7 --do_recall

  python bench_vec_cache_full.py --metric ip --use_ivfpq --n_list 4096 --nprobe 16 --pq_m 32 --pq_nbits 8
"""

import argparse
import time
import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("faiss is required (pip install faiss-cpu or faiss-gpu).") from e


def set_threads(n: int):
    try:
        faiss.omp_set_num_threads(n)
    except Exception:
        pass


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X32 = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X32, axis=1, keepdims=True) + 1e-12
    return (X32 / norms).astype(X.dtype, copy=False)


def pcts(lat_ms: np.ndarray):
    return float(np.percentile(lat_ms, 50)), float(np.percentile(lat_ms, 95))


def l2_topk_bruteforce(X: np.ndarray, Q: np.ndarray, k: int, batch: int):
    """
    Exact brute-force top-k for L2: returns (I, D2) where D2 is squared L2 distance.
    Uses: ||x-q||^2 = ||x||^2 + ||q||^2 - 2 x·q
    """
    X32 = X.astype(np.float32, copy=False)
    Q32 = Q.astype(np.float32, copy=False)
    x_norm = (X32 * X32).sum(axis=1, keepdims=True)  # (N,1)

    I_all, D_all = [], []
    for i in range(0, Q32.shape[0], batch):
        q = Q32[i:i + batch]  # (b,d)
        q_norm = (q * q).sum(axis=1, keepdims=True).T  # (1,b)
        dist2 = x_norm + q_norm - 2.0 * (X32 @ q.T)    # (N,b)

        idx = np.argpartition(dist2, kth=k - 1, axis=0)[:k, :]  # (k,b)
        dsel = np.take_along_axis(dist2, idx, axis=0)           # (k,b)

        order = np.argsort(dsel, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)
        dsel = np.take_along_axis(dsel, order, axis=0)

        I_all.append(idx.T.copy())    # (b,k)
        D_all.append(dsel.T.copy())   # (b,k)

    return np.vstack(I_all), np.vstack(D_all)


def ip_topk_bruteforce(X: np.ndarray, Q: np.ndarray, k: int, batch: int):
    """
    Exact brute-force top-k for inner product: returns (I, S) where larger S is better.
    """
    X32 = X.astype(np.float32, copy=False)
    Q32 = Q.astype(np.float32, copy=False)

    I_all, S_all = [], []
    for i in range(0, Q32.shape[0], batch):
        q = Q32[i:i + batch]    # (b,d)
        scores = X32 @ q.T      # (N,b)

        # Get top-k largest per column
        idx = np.argpartition(scores, kth=X32.shape[0] - k, axis=0)[-k:, :]  # (k,b)
        ssel = np.take_along_axis(scores, idx, axis=0)

        order = np.argsort(-ssel, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)
        ssel = np.take_along_axis(ssel, order, axis=0)

        I_all.append(idx.T.copy())   # (b,k)
        S_all.append(ssel.T.copy())  # (b,k)

    return np.vstack(I_all), np.vstack(S_all)


def time_bruteforce(X: np.ndarray, Q: np.ndarray, metric: str, k: int, batch: int, reps: int, warmup: int):
    lat = []

    # warmup
    for _ in range(warmup):
        q0 = Q[: min(batch, Q.shape[0])]
        if metric == "l2":
            l2_topk_bruteforce(X, q0, k, batch=batch)
        else:
            ip_topk_bruteforce(X, q0, k, batch=batch)

    # timed
    for _ in range(reps):
        t0 = time.perf_counter()
        if metric == "l2":
            l2_topk_bruteforce(X, Q, k, batch=batch)
        else:
            ip_topk_bruteforce(X, Q, k, batch=batch)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0 / Q.shape[0])

    lat_ms = np.array(lat, dtype=np.float64)
    return pcts(lat_ms), lat_ms


def time_faiss(index, Q: np.ndarray, k: int, reps: int, warmup: int):
    lat = []
    Q32 = Q.astype(np.float32, copy=False)

    # warmup
    for _ in range(warmup):
        index.search(Q32[: min(16, Q32.shape[0])], k)

    for _ in range(reps):
        t0 = time.perf_counter()
        index.search(Q32, k)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0 / Q32.shape[0])

    lat_ms = np.array(lat, dtype=np.float64)
    return pcts(lat_ms), lat_ms


def build_faiss_hnsw(X: np.ndarray, metric: str, efC: int, efS: int, hnsw_m: int):
    d = X.shape[1]
    if metric == "l2":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_L2)
    else:
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)

    index.hnsw.efConstruction = efC
    index.hnsw.efSearch = efS
    index.add(X.astype(np.float32, copy=False))
    return index


def build_faiss_ivfflat(X: np.ndarray, metric: str, nlist: int):
    d = X.shape[1]
    if metric == "l2":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    X32 = X.astype(np.float32, copy=False)
    index.train(X32)
    index.add(X32)
    return index


def build_faiss_ivfpq(X: np.ndarray, metric: str, nlist: int, m: int, nbits: int):
    d = X.shape[1]
    if metric == "l2":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)
    else:
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    X32 = X.astype(np.float32, copy=False)
    index.train(X32)
    index.add(X32)
    return index


def recall_at_k(I_exact: np.ndarray, I_test: np.ndarray, k: int) -> float:
    # average fraction of overlap with exact top-k
    hits = 0
    for a, b in zip(I_exact, I_test):
        hits += len(set(a[:k]).intersection(set(b[:k])))
    return hits / (I_exact.shape[0] * k)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--metric", choices=["ip", "l2", "cosine"], default="ip")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--q", type=int, default=200, help="number of queries per N")
    ap.add_argument("--min_n", type=int, default=50_000)
    ap.add_argument("--max_n", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=6, help="log-spaced steps between min_n and max_n")

    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16",
                    help="storage dtype for brute-force matrix")
    ap.add_argument("--batch", type=int, default=16, help="brute-force query batch")
    ap.add_argument("--threads", type=int, default=0, help="FAISS/OMP threads (0=default)")

    ap.add_argument("--use_hnsw", action="store_true", default=False)
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=128)

    ap.add_argument("--use_ivfflat", action="store_true", default=False)
    ap.add_argument("--ivf_nlist", type=int, default=4096)
    ap.add_argument("--ivf_nprobe", type=int, default=16)

    ap.add_argument("--use_ivfpq", action="store_true", default=True)
    ap.add_argument("--n_list", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--pq_m", type=int, default=32)
    ap.add_argument("--pq_nbits", type=int, default=8)

    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)

    ap.add_argument("--do_recall", action="store_true",
                    help="compute Recall@K for ANN baselines vs exact brute force (slower)")
    args = ap.parse_args()

    if args.threads > 0:
        set_threads(args.threads)

    # internal metric
    metric = args.metric
    if args.metric == "cosine":
        metric = "ip"  # cosine = IP over normalized vectors

    # log-spaced N sweep
    Ns = np.unique(np.round(np.exp(np.linspace(np.log(args.min_n), np.log(args.max_n), args.steps))).astype(int))

    print(f"Benchmark: d={args.d} metric={args.metric} (internal={metric}) k={args.k} q={args.q}")
    print(f"Brute-force: storage={args.dtype} batch={args.batch}")
    if args.use_hnsw:
        print(f"HNSW: M={args.hnsw_m} efC={args.efC} efS={args.efS}")
    if args.use_ivfflat:
        print(f"IVF-Flat: nlist={args.ivf_nlist} nprobe={args.ivf_nprobe}")
    if args.use_ivfpq:
        print(f"IVF-PQ: nlist={args.n_list} nprobe={args.nprobe} m={args.pq_m} nbits={args.pq_nbits}")
    print()

    rng = np.random.default_rng(0)

    for N in Ns:
        # synthetic data
        X = rng.standard_normal((N, args.d), dtype=np.float32)
        Q = rng.standard_normal((args.q, args.d), dtype=np.float32)

        if args.metric == "cosine":
            X = normalize_rows(X)
            Q = normalize_rows(Q)

        # brute-force storage matrix
        X_store = X.astype(np.float16) if args.dtype == "float16" else X.astype(np.float32)

        # exact top-k for recall
        I_exact = None
        if args.do_recall:
            if metric == "l2":
                I_exact, _ = l2_topk_bruteforce(X.astype(np.float32), Q.astype(np.float32), args.k, batch=max(1, args.batch))
            else:
                I_exact, _ = ip_topk_bruteforce(X.astype(np.float32), Q.astype(np.float32), args.k, batch=max(1, args.batch))

        # brute-force timing
        (bf_p50, bf_p95), _ = time_bruteforce(
            X_store, Q, metric=metric, k=args.k, batch=max(1, args.batch),
            reps=args.reps, warmup=args.warmup
        )

        line = f"N={N:>9,d} | brute_force p50={bf_p50:7.3f} ms p95={bf_p95:7.3f} ms"

        # HNSW
        if args.use_hnsw:
            index_h = build_faiss_hnsw(X, metric=metric, efC=args.efC, efS=args.efS, hnsw_m=args.hnsw_m)
            (h_p50, h_p95), _ = time_faiss(index_h, Q, k=args.k, reps=args.reps, warmup=args.warmup)
            line += f" | hnsw p50={h_p50:7.3f} ms p95={h_p95:7.3f} ms"
            if args.do_recall:
                I_h, _ = index_h.search(Q.astype(np.float32), args.k)
                line += f" R@{args.k}={recall_at_k(I_exact, I_h, args.k):.3f}"

        # IVF-Flat
        if args.use_ivfflat:
            index_ivf = build_faiss_ivfflat(X, metric=metric, nlist=args.ivf_nlist)
            index_ivf.nprobe = args.ivf_nprobe
            (ivf_p50, ivf_p95), _ = time_faiss(index_ivf, Q, k=args.k, reps=args.reps, warmup=args.warmup)
            line += f" | ivfflat p50={ivf_p50:7.3f} ms p95={ivf_p95:7.3f} ms"
            if args.do_recall:
                I_ivf, _ = index_ivf.search(Q.astype(np.float32), args.k)
                line += f" R@{args.k}={recall_at_k(I_exact, I_ivf, args.k):.3f}"

        # IVF-PQ
        if args.use_ivfpq:
            index_pq = build_faiss_ivfpq(X, metric=metric, nlist=args.n_list, m=args.pq_m, nbits=args.pq_nbits)
            index_pq.nprobe = args.nprobe
            (pq_p50, pq_p95), _ = time_faiss(index_pq, Q, k=args.k, reps=args.reps, warmup=args.warmup)
            line += f" | ivfpq p50={pq_p50:7.3f} ms p95={pq_p95:7.3f} ms"
            if args.do_recall:
                I_pq, _ = index_pq.search(Q.astype(np.float32), args.k)
                line += f" R@{args.k}={recall_at_k(I_exact, I_pq, args.k):.3f}"

        print(line)


if __name__ == "__main__":
    main()