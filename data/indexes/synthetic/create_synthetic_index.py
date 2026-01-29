#!/usr/bin/env python3
import os
import csv
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import faiss


# -------------------------
# Utilities
# -------------------------

def bytes_to_human(num_bytes: int) -> str:
    x = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024.0 or unit == "TB":
            return f"{x:.2f}{unit}"
        x /= 1024.0
    return f"{x:.2f}TB"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# Synthetic data generation
# -------------------------

def make_vectors(n: int, d: int = 768, seed: int = 123, normalize: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n, d)).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = x / norms
    return x


# -------------------------
# FAISS index building
# -------------------------

@dataclass
class IndexConfig:
    index_type: str            # "flat" or "hnsw" or "ivf"
    metric: str                # "ip" or "l2"
    d: int
    n: int
    normalize: bool

    # HNSW params (optional)
    hnsw_m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None

    # IVF params (optional)
    ivf_nlist: Optional[int] = None
    ivf_nprobe: Optional[int] = None
    ivf_train_size: Optional[int] = None  # how many vectors to use for training


def _faiss_metric(metric: str) -> int:
    metric = metric.lower()
    if metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric == "l2":
        return faiss.METRIC_L2
    raise ValueError("metric must be 'ip' or 'l2'")


def build_index(vectors: np.ndarray, cfg: IndexConfig):
    """
    Build a FAISS index according to cfg and add vectors.
    Returns: (index, add_time_sec, train_time_sec)
    """
    n, d = vectors.shape
    assert d == cfg.d and n == cfg.n, "Vector shape does not match config."

    m = _faiss_metric(cfg.metric)
    train_time = 0.0

    if cfg.index_type == "flat":
        index = faiss.IndexFlatIP(d) if m == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)

    elif cfg.index_type == "hnsw":
        if cfg.hnsw_m is None or cfg.ef_construction is None or cfg.ef_search is None:
            raise ValueError("HNSW requires hnsw_m, ef_construction, ef_search.")
        index = faiss.IndexHNSWFlat(d, int(cfg.hnsw_m), m)
        index.hnsw.efConstruction = int(cfg.ef_construction)
        index.hnsw.efSearch = int(cfg.ef_search)

    elif cfg.index_type == "ivf":
        # IVF requires training
        if cfg.ivf_nlist is None:
            raise ValueError("IVF requires ivf_nlist.")
        # nprobe is query-time param, but we store it for reproducibility
        nlist = int(cfg.ivf_nlist)

        quantizer = faiss.IndexFlatIP(d) if m == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, m)

        train_size = int(cfg.ivf_train_size) if cfg.ivf_train_size is not None else min(n, max(10_000, nlist * 50))
        train_size = min(train_size, n)

        # Train on a subset (prefix is fine for synthetic; for real data, sample)
        t0 = time.perf_counter()
        index.train(vectors[:train_size])
        t1 = time.perf_counter()
        train_time = t1 - t0

        if cfg.ivf_nprobe is not None:
            index.nprobe = int(cfg.ivf_nprobe)

    else:
        raise ValueError("index_type must be 'flat', 'hnsw', or 'ivf'")

    t0 = time.perf_counter()
    index.add(vectors)
    t1 = time.perf_counter()
    add_time = t1 - t0

    return index, add_time, train_time


def index_filename(cfg: IndexConfig) -> str:
    base = f"{cfg.index_type}_{cfg.metric}_d{cfg.d}_n{cfg.n}_norm{int(cfg.normalize)}"
    if cfg.index_type == "hnsw":
        base += f"_m{cfg.hnsw_m}_efc{cfg.ef_construction}_efs{cfg.ef_search}"
    if cfg.index_type == "ivf":
        base += f"_nlist{cfg.ivf_nlist}_nprobe{cfg.ivf_nprobe}_train{cfg.ivf_train_size}"
    return base + ".index"


def save_index(index: faiss.Index, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    faiss.write_index(index, path)


def serialized_size_bytes(index: faiss.Index) -> int:
    return len(faiss.serialize_index(index))


# -------------------------
# Manifest (CSV)
# -------------------------

MANIFEST_FIELDS = [
    "created_at_unix",
    "index_path",
    "index_type",
    "metric",
    "d",
    "n",
    "normalize",
    # HNSW
    "hnsw_m",
    "ef_construction",
    "ef_search",
    # IVF
    "ivf_nlist",
    "ivf_nprobe",
    "ivf_train_size",
    # sizes/timing
    "raw_vector_bytes_est",
    "serialized_bytes_est",
    "disk_bytes",
    "train_time_sec",
    "add_time_sec",
]


def append_manifest_row(csv_path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(csv_path)
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in MANIFEST_FIELDS})


# -------------------------
# Main driver
# -------------------------

def build_indexes_and_manifest(
    out_dir: str,
    manifest_csv: str,
    vector_counts: List[int],
    d: int = 768,
    metric: str = "ip",
    normalize: bool = True,
    seed: int = 123,
    build_flat: bool = True,
    build_hnsw: bool = True,
    build_ivf: bool = True,
    # HNSW
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    # IVF
    ivf_nlist_factor: float = 1.0,   # nlist ~ factor * sqrt(N)
    ivf_nprobe: int = 16,
    ivf_train_size: Optional[int] = None,
) -> None:
    ensure_dir(out_dir)

    metric = metric.lower()
    if metric not in ("ip", "l2"):
        raise ValueError("metric must be 'ip' or 'l2'")

    for n in vector_counts:
        print(f"\n=== N={n:,}  d={d}  metric={metric}  normalize={normalize} ===")
        raw_bytes = n * d * 4
        print(f"Raw vectors (float32) estimate: {bytes_to_human(raw_bytes)}")

        xb = make_vectors(n, d=d, seed=seed, normalize=normalize)

        configs: List[IndexConfig] = []
        if build_flat:
            configs.append(IndexConfig(index_type="flat", metric=metric, d=d, n=n, normalize=normalize))

        if build_hnsw:
            configs.append(IndexConfig(
                index_type="hnsw", metric=metric, d=d, n=n, normalize=normalize,
                hnsw_m=hnsw_m, ef_construction=ef_construction, ef_search=ef_search
            ))

        if build_ivf:
            nlist = max(1, int(ivf_nlist_factor * np.sqrt(n)))
            train_sz = ivf_train_size if ivf_train_size is not None else min(n, max(10_000, nlist * 50))
            configs.append(IndexConfig(
                index_type="ivf", metric=metric, d=d, n=n, normalize=normalize,
                ivf_nlist=nlist, ivf_nprobe=ivf_nprobe, ivf_train_size=train_sz
            ))

        for cfg in configs:
            fname = index_filename(cfg)
            path = os.path.join(out_dir, fname)

            print(f"  -> Building {cfg.index_type.upper()}  ({fname})")
            index, add_time, train_time = build_index(xb, cfg)

            ser_bytes = serialized_size_bytes(index)
            save_index(index, path)
            disk_bytes = os.path.getsize(path)

            extra = ""
            if cfg.index_type == "ivf":
                extra = f"  train_time={train_time:.2f}s nlist={cfg.ivf_nlist} nprobe={cfg.ivf_nprobe}"

            print(
                f"     add_time={add_time:.2f}s{extra}  "
                f"serialized≈{bytes_to_human(ser_bytes)}  "
                f"disk={bytes_to_human(disk_bytes)}"
            )

            append_manifest_row(manifest_csv, {
                "created_at_unix": int(time.time()),
                "index_path": path,
                "index_type": cfg.index_type,
                "metric": cfg.metric,
                "d": cfg.d,
                "n": cfg.n,
                "normalize": int(cfg.normalize),

                "hnsw_m": cfg.hnsw_m,
                "ef_construction": cfg.ef_construction,
                "ef_search": cfg.ef_search,

                "ivf_nlist": cfg.ivf_nlist,
                "ivf_nprobe": cfg.ivf_nprobe,
                "ivf_train_size": cfg.ivf_train_size,

                "raw_vector_bytes_est": f"{raw_bytes} ({bytes_to_human(raw_bytes)})",
                "serialized_bytes_est": f"{ser_bytes} ({bytes_to_human(ser_bytes)})",
                "disk_bytes": f"{disk_bytes} ({bytes_to_human(disk_bytes)})",
                "train_time_sec": round(train_time, 6),
                "add_time_sec": round(add_time, 6),
            })

        del xb


if __name__ == "__main__":
    OUT_DIR = "data/indexes/synthetic"
    MANIFEST_CSV = "data/indexes/synthetic/synthetic_index_manifest.csv"

    D = 768
    METRIC = "ip"
    NORMALIZE = True
    SEED = 123

    VECTOR_COUNTS = [
        # 100_000,
        # 500_000,
        1_000_000,
    ]

    BUILD_FLAT = True
    BUILD_HNSW = True
    BUILD_IVF = True

    # HNSW params
    HNSW_M = 32
    EF_CONSTRUCTION = 200
    EF_SEARCH = 64

    # IVF params (good defaults to start)
    IVF_NLIST_FACTOR = 1.0   # nlist ≈ sqrt(N)
    IVF_NPROBE = 16          # search-time knob (bigger = better recall, slower)
    IVF_TRAIN_SIZE = None    # None => auto (max(10k, 50*nlist))

    build_indexes_and_manifest(
        out_dir=OUT_DIR,
        manifest_csv=MANIFEST_CSV,
        vector_counts=VECTOR_COUNTS,
        d=D,
        metric=METRIC,
        normalize=NORMALIZE,
        seed=SEED,
        build_flat=BUILD_FLAT,
        build_hnsw=BUILD_HNSW,
        build_ivf=BUILD_IVF,
        hnsw_m=HNSW_M,
        ef_construction=EF_CONSTRUCTION,
        ef_search=EF_SEARCH,
        ivf_nlist_factor=IVF_NLIST_FACTOR,
        ivf_nprobe=IVF_NPROBE,
        ivf_train_size=IVF_TRAIN_SIZE,
    )

    print(f"\nDone. Manifest written to: {MANIFEST_CSV}")
