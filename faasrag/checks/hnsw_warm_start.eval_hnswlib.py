import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import hnswlib

# -----------------------------
# Utilities
# -----------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))

def brute_force_topk_cosine(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    sims = base @ query
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    return idx[np.argsort(-sims[idx])]

def evaluate_recall_at_k(approx_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    return float(len(set(approx_ids.tolist()) & set(gt_ids.tolist())) / len(gt_ids))


# -----------------------------
# Config
# -----------------------------

@dataclass
class ExperimentConfig:
    # data
    N: int = 200_000
    d: int = 768

    # HNSW params
    M: int = 16
    ef_construction: int = 200
    ef_search_base: int = 80         # ef for baseline
    ef_search_warm: int = 50         # ef for warm-started queries
    ef_search_cold: int = 80         # ef for non-warm queries in "all-warm" mode

    # eval
    n_queries: int = 5_000
    k: int = 10

    # correlated query generation
    similar_prob: float = 0.6
    noise_sigma: float = 0.01

    # warm-start gating
    gate_cosine_tau: float = 0.95
    seed_M: int = 8

    # brute-force GT subset
    gt_sample: int = 50_000

    # random
    seed: int = 123


# -----------------------------
# HNSW helpers
# -----------------------------

class HNSWRunner:
    def __init__(self, index: hnswlib.Index, cfg: ExperimentConfig):
        self.index = index
        self.cfg = cfg

    def knn_query_baseline(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        labels, dists = self.index.knn_query(q, k=self.cfg.k)
        return labels[0], dists[0]

    def knn_query_seeded(self, q: np.ndarray, seeds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # IMPORTANT: your binding is (data, seed_ids, k=..., num_threads=..., filter=...)
        labels, dists = self.index.knn_query_seeded(q, seeds, k=self.cfg.k)
        return labels[0], dists[0]


def build_hnsw_index(base: np.ndarray, cfg: ExperimentConfig) -> hnswlib.Index:
    p = hnswlib.Index(space="cosine", dim=cfg.d)
    p.init_index(max_elements=cfg.N, ef_construction=cfg.ef_construction, M=cfg.M)
    p.add_items(base, np.arange(cfg.N))
    p.set_ef(cfg.ef_search_base)
    return p


def generate_correlated_queries(base_queries: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 999)
    Q = base_queries.copy()
    for i in range(1, Q.shape[0]):
        if rng.random() < cfg.similar_prob:
            noise = rng.normal(0, cfg.noise_sigma, size=cfg.d).astype(np.float32)
            Q[i] = Q[i - 1] + noise
    return l2_normalize(Q.astype(np.float32))


# -----------------------------
# Experiment
# -----------------------------

def summarize(name: str, lat: np.ndarray, rec: np.ndarray, k: int):
    print(f"\n=== {name} ===")
    if lat.size == 0:
        print("No samples.")
        return
    print(f"Latency (ms): p50={percentile(lat, 50):.3f}  p95={percentile(lat, 95):.3f}  p99={percentile(lat, 99):.3f}")
    print(f"Recall@{k}:   mean={rec.mean():.4f}  p50={percentile(rec, 50):.4f}  p95={percentile(rec, 95):.4f}")


def run_experiment(cfg: ExperimentConfig):
    print("hnswlib path:", hnswlib.__file__)
    print("Has knn_query_seeded:", hasattr(hnswlib.Index, "knn_query_seeded"))

    rng = np.random.default_rng(cfg.seed)

    # 1) Base vectors
    base = rng.normal(size=(cfg.N, cfg.d)).astype(np.float32)
    base = l2_normalize(base)

    # 2) Build index
    index = build_hnsw_index(base, cfg)
    runner = HNSWRunner(index, cfg)

    # 3) Query stream
    Q0 = rng.normal(size=(cfg.n_queries, cfg.d)).astype(np.float32)
    Q0 = l2_normalize(Q0)
    queries = generate_correlated_queries(Q0, cfg)

    # 4) GT subset
    gt_idx = rng.choice(cfg.N, size=min(cfg.gt_sample, cfg.N), replace=False)
    base_gt = base[gt_idx]

    # Storage
    lat_base = np.zeros(cfg.n_queries, dtype=np.float64)
    rec_base = np.zeros(cfg.n_queries, dtype=np.float64)

    lat_warm_all = np.zeros(cfg.n_queries, dtype=np.float64)   # "all queries warm" mode
    rec_warm_all = np.zeros(cfg.n_queries, dtype=np.float64)

    use_warm = np.zeros(cfg.n_queries, dtype=bool)

    cached_seed_ids: Optional[np.ndarray] = None
    cached_query: Optional[np.ndarray] = None

    gate_hits = 0
    seeded_calls = 0

    # Main loop
    for i in range(cfg.n_queries):
        q = queries[i]

        # Decide whether warm-start applies for this query
        warm = False
        if cached_seed_ids is not None and cached_query is not None:
            warm = cosine_sim(cached_query, q) >= cfg.gate_cosine_tau
        use_warm[i] = warm
        if warm:
            gate_hits += 1

        # ---- Baseline (all queries, fixed ef_search_base)
        index.set_ef(cfg.ef_search_base)
        t0 = time.perf_counter()
        ids_b, _ = runner.knn_query_baseline(q)
        lat_base[i] = (time.perf_counter() - t0) * 1e3

        gt = brute_force_topk_cosine(base_gt, q, cfg.k)
        gt_ids = gt_idx[gt]
        rec_base[i] = evaluate_recall_at_k(ids_b, gt_ids)

        # ---- Warm (all queries)
        # If warm-hit: run seeded + ef_search_warm
        # Else: run baseline search (no seeds) but possibly different ef_search_cold
        if warm:
            index.set_ef(cfg.ef_search_warm)
            t1 = time.perf_counter()
            seeded_calls += 1
            ids_w, _ = runner.knn_query_seeded(q, cached_seed_ids)
            lat_warm_all[i] = (time.perf_counter() - t1) * 1e3
            rec_warm_all[i] = evaluate_recall_at_k(ids_w, gt_ids)
        else:
            # "all queries warm" means you still run something for every query.
            # Here we run normal search with ef_search_cold.
            index.set_ef(cfg.ef_search_cold)
            t2 = time.perf_counter()
            ids_c, _ = runner.knn_query_baseline(q)
            lat_warm_all[i] = (time.perf_counter() - t2) * 1e3
            rec_warm_all[i] = evaluate_recall_at_k(ids_c, gt_ids)

        # Update cache from THIS query (cache baseline top ids)
        cached_seed_ids = ids_b[: cfg.seed_M].astype(np.int64)
        cached_query = q.copy()

    # Reporting
    print(f"\nGate hits: {gate_hits}/{cfg.n_queries}")
    print(f"Seeded calls executed: {seeded_calls}")

    summarize("Baseline (all queries)", lat_base, rec_base, cfg.k)
    summarize("Warm system (all queries: seeded on hits)", lat_warm_all, rec_warm_all, cfg.k)

    # Paired subset analysis on the same warm-hit queries
    mask = use_warm
    summarize("Baseline (warm-hit subset)", lat_base[mask], rec_base[mask], cfg.k)
    summarize("Warm (warm-hit subset)", lat_warm_all[mask], rec_warm_all[mask], cfg.k)

    # Optional: cold subset too
    cold = ~use_warm
    summarize("Baseline (cold subset)", lat_base[cold], rec_base[cold], cfg.k)
    summarize("Warm system (cold subset)", lat_warm_all[cold], rec_warm_all[cold], cfg.k)


if __name__ == "__main__":
    cfg = ExperimentConfig(
        N=200_000,
        d=768,
        M=16,
        ef_construction=200,
        ef_search_base=80,
        ef_search_warm=50,   # try 60/50/40/30 sweep
        ef_search_cold=80,   # keep cold same as baseline initially
        n_queries=5000,
        k=10,
        similar_prob=0.6,
        noise_sigma=0.01,
        gate_cosine_tau=0.95,
        seed_M=8,
        gt_sample=50_000,
        seed=123,
    )
    run_experiment(cfg)
