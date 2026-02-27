#!/usr/bin/env python3
"""
Microbenchmark for a HOT-TIER vector cache + Tier-3 ANN fallback.

What it measures:
- Tier 1 (hot tier): exact brute-force search over a hot subset matrix (size sweeps).
- Tier 3 (full tier): FAISS ANN search over the full dataset (fixed size).

It reports:
- Tier 1 p50/p95 per-query latency
- Tier 3 p50/p95 per-query latency
- Simulated end-to-end latency for a set of hit rates:
    E[lat] = t_hot + (1-hit_rate)*t_tier3

Optional:
- Recall@K for Tier-3 ANN vs exact brute force over FULL dataset (can be expensive).

Notes:
- This is CPU-only brute force (NumPy). FAISS ANN can be CPU (faiss-cpu) or GPU (faiss-gpu),
  but this script uses CPU indices by default.
- For cosine: normalize vectors and use inner product.

Examples:
  python bench_hot_tier.py --metric cosine --full_n 2000000 --q 200 --k 10 \
    --use_ivfpq --nprobe 32 --pq_m 32 --pq_nbits 8 \
    --hot_min_n 10000 --hot_max_n 200000 --hot_steps 6 --dtype float16

  python bench_hot_tier.py --metric ip --full_n 5000000 --q 200 --k 10 \
    --use_ivfflat --ivf_nprobe 32 \
    --hot_min_n 20000 --hot_max_n 500000 --hot_steps 6 --dtype float16

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


def ip_topk_bruteforce(X: np.ndarray, Q: np.ndarray, k: int, batch: int):
    X32 = X.astype(np.float32, copy=False)
    Q32 = Q.astype(np.float32, copy=False)
    I_all, S_all = [], []

    for i in range(0, Q32.shape[0], batch):
        q = Q32[i:i + batch]          # (b,d)
        scores = X32 @ q.T            # (N,b)

        idx = np.argpartition(scores, kth=X32.shape[0] - k, axis=0)[-k:, :]  # (k,b)
        ssel = np.take_along_axis(scores, idx, axis=0)

        order = np.argsort(-ssel, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)
        ssel = np.take_along_axis(ssel, order, axis=0)

        I_all.append(idx.T.copy())    # (b,k)
        S_all.append(ssel.T.copy())   # (b,k)

    return np.vstack(I_all), np.vstack(S_all)


def l2_topk_bruteforce(X: np.ndarray, Q: np.ndarray, k: int, batch: int):
    X32 = X.astype(np.float32, copy=False)
    Q32 = Q.astype(np.float32, copy=False)
    x_norm = (X32 * X32).sum(axis=1, keepdims=True)  # (N,1)

    I_all, D_all = [], []
    for i in range(0, Q32.shape[0], batch):
        q = Q32[i:i + batch]  # (b,d)
        q_norm = (q * q).sum(axis=1, keepdims=True).T  # (1,b)
        dist2 = x_norm + q_norm - 2.0 * (X32 @ q.T)    # (N,b)

        idx = np.argpartition(dist2, kth=k - 1, axis=0)[:k, :]  # (k,b)
        dsel = np.take_along_axis(dist2, idx, axis=0)

        order = np.argsort(dsel, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)
        dsel = np.take_along_axis(dsel, order, axis=0)

        I_all.append(idx.T.copy())    # (b,k)
        D_all.append(dsel.T.copy())   # (b,k)

    return np.vstack(I_all), np.vstack(D_all)


def time_bruteforce(X: np.ndarray, Q: np.ndarray, metric: str, k: int, batch: int, reps: int, warmup: int):
    lat = []
    q0 = Q[: min(batch, Q.shape[0])]

    for _ in range(warmup):
        if metric == "l2":
            l2_topk_bruteforce(X, q0, k, batch=batch)
        else:
            ip_topk_bruteforce(X, q0, k, batch=batch)

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
    hits = 0
    for a, b in zip(I_exact, I_test):
        hits += len(set(a[:k]).intersection(set(b[:k])))
    return hits / (I_exact.shape[0] * k)


def default_nlist(full_n: int) -> int:
    # Rough heuristic: about sqrt(N), rounded to power-of-two-ish.
    base = int(np.sqrt(full_n))
    # clamp
    base = max(64, min(base, 65536))
    # round to nearest power of 2
    p = 1 << int(np.round(np.log2(base)))
    return int(max(64, min(p, 65536)))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--metric", choices=["ip", "l2", "cosine"], default="ip")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--q", type=int, default=200, help="number of queries")

    ap.add_argument("--full_n", type=int, default=1_000_000, help="Tier 3 full dataset size (fixed)")

    ap.add_argument("--hot_min_n", type=int, default=10_000)
    ap.add_argument("--hot_max_n", type=int, default=200_000)
    ap.add_argument("--hot_steps", type=int, default=6, help="log-spaced steps for hot tier size sweep")

    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16", help="hot tier storage dtype")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)

    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)

    ap.add_argument("--hit_rates", type=str, default="0.3,0.5,0.7,0.9",
                    help="comma-separated hit rates to simulate for end-to-end latency")

    # ANN choices
    ap.add_argument("--use_hnsw", action="store_true", default=False)
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--efC", type=int, default=200)
    ap.add_argument("--efS", type=int, default=128)

    ap.add_argument("--use_ivfflat", action="store_true", default=False)
    ap.add_argument("--ivf_nlist", type=int, default=0, help="0 => auto")
    ap.add_argument("--ivf_nprobe", type=int, default=32)

    ap.add_argument("--use_ivfpq", action="store_true", default=True)
    ap.add_argument("--n_list", type=int, default=0, help="0 => auto")
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--pq_m", type=int, default=32)
    ap.add_argument("--pq_nbits", type=int, default=8)

    ap.add_argument("--do_recall", action="store_true",
                    help="compute Recall@K for Tier3 ANN vs exact brute force over FULL dataset (slower)")
    args = ap.parse_args()

    if args.threads > 0:
        set_threads(args.threads)

    # internal metric
    metric = args.metric
    if args.metric == "cosine":
        metric = "ip"

    # Parse hit rates
    hit_rates = [float(x.strip()) for x in args.hit_rates.split(",") if x.strip()]
    hit_rates = [min(1.0, max(0.0, h)) for h in hit_rates]

    # Hot sweep Ns
    hot_Ns = np.unique(np.round(np.exp(np.linspace(np.log(args.hot_min_n), np.log(args.hot_max_n), args.hot_steps))).astype(int))

    print(f"Tiered benchmark: d={args.d} metric={args.metric} (internal={metric}) k={args.k} q={args.q}")
    print(f"Tier3(full) N={args.full_n:,} | Hot sweep: {hot_Ns[0]:,}..{hot_Ns[-1]:,} ({len(hot_Ns)} steps)")
    print(f"Hot storage dtype={args.dtype} batch={args.batch}")
    print(f"Simulated hit rates: {hit_rates}")
    print()

    rng = np.random.default_rng(0)

    # Build FULL dataset and queries once (fixed)
    X_full = rng.standard_normal((args.full_n, args.d), dtype=np.float32)
    Q = rng.standard_normal((args.q, args.d), dtype=np.float32)

    if args.metric == "cosine":
        X_full = normalize_rows(X_full)
        Q = normalize_rows(Q)

    # Build Tier-3 ANN index once
    tier3_name = None
    tier3_index = None

    if args.use_hnsw:
        tier3_name = "hnsw"
        tier3_index = build_faiss_hnsw(X_full, metric=metric, efC=args.efC, efS=args.efS, hnsw_m=args.hnsw_m)

    elif args.use_ivfflat:
        tier3_name = "ivfflat"
        nlist = args.ivf_nlist if args.ivf_nlist > 0 else default_nlist(args.full_n)
        tier3_index = build_faiss_ivfflat(X_full, metric=metric, nlist=nlist)
        tier3_index.nprobe = args.ivf_nprobe
        print(f"Tier3 IVF-Flat config: nlist={nlist} nprobe={args.ivf_nprobe}")

    else:
        # default: ivfpq
        tier3_name = "ivfpq"
        nlist = args.n_list if args.n_list > 0 else default_nlist(args.full_n)
        tier3_index = build_faiss_ivfpq(X_full, metric=metric, nlist=nlist, m=args.pq_m, nbits=args.pq_nbits)
        tier3_index.nprobe = args.nprobe
        print(f"Tier3 IVF-PQ config: nlist={nlist} nprobe={args.nprobe} m={args.pq_m} nbits={args.pq_nbits}")

    # Time Tier3 once
    (t3_p50, t3_p95), _ = time_faiss(tier3_index, Q, k=args.k, reps=args.reps, warmup=args.warmup)
    print(f"Tier3 {tier3_name} latency: p50={t3_p50:.3f} ms  p95={t3_p95:.3f} ms")
    print()

    # Optional: recall for Tier3 vs exact brute force over FULL dataset
    if args.do_recall:
        if metric == "l2":
            I_exact, _ = l2_topk_bruteforce(X_full, Q, args.k, batch=max(1, args.batch))
        else:
            I_exact, _ = ip_topk_bruteforce(X_full, Q, args.k, batch=max(1, args.batch))
        I_t3, _ = tier3_index.search(Q.astype(np.float32), args.k)
        r = recall_at_k(I_exact, I_t3, args.k)
        print(f"Tier3 Recall@{args.k} vs exact(full): {r:.3f}")
        print()

    # Hot tier sweep: select hot subset as first N_hot vectors (easy + deterministic)
    # In a real system you'd select most popular vectors; for benchmarking compute, any subset is fine.
    for hot_n in hot_Ns:
        X_hot = X_full[:hot_n]  # subset view
        X_hot_store = X_hot.astype(np.float16) if args.dtype == "float16" else X_hot.astype(np.float32)

        (h_p50, h_p95), _ = time_bruteforce(
            X_hot_store, Q, metric=metric, k=args.k, batch=max(1, args.batch),
            reps=args.reps, warmup=args.warmup
        )

        # Simulated end-to-end: always pay hot tier + miss pay Tier3
        sims = []
        for hr in hit_rates:
            sims.append(h_p50 + (1.0 - hr) * t3_p50)
        sims_str = " ".join([f"hr={hr:.2f}->E[p50]={val:.3f}ms" for hr, val in zip(hit_rates, sims)])

        print(f"HotN={hot_n:>9,d} | hot p50={h_p50:7.3f} ms p95={h_p95:7.3f} ms | "
              f"tier3({tier3_name}) p50={t3_p50:7.3f} ms | {sims_str}")


if __name__ == "__main__":
    main()