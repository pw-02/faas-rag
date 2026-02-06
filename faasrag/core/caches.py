from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import proximipy

from faasrag.core.args import CacheConfig, ProximityCacheConfig


class ProximityCache:
    """
    Wrapper around proximipy caches with optional read-through caching for ANN search.

    Notes:
    - For LSH policies ("lsh_*"), you MUST provide embedding dim.
    - Keys passed to proximipy are expected to be list[float] (or similar).
    """

    def __init__(
        self,
        *,
        policy: str,
        tolerance: float,
        capacity: int,
        lsh_bucket_capacity: Optional[int] = None,
        lsh_num_hashes: Optional[int] = None,
        seed: Optional[int] = None,
        dim: Optional[int] = None,
    ):
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        self.tolerance = float(tolerance)
        self.capacity = int(capacity)
        self.seed = seed

        # Normalize policy names (support legacy aliases)
        p = str(policy).lower()
        p = {"lpt": "lru", "lsh_lpt": "lsh_lru"}.get(p, p)

        if p not in {"fifo", "lru", "lsh_fifo", "lsh_lru"}:
            raise ValueError(f"Unknown policy={policy!r} (normalized={p!r})")

        self.policy = p
        self.dim = dim

        self.lsh_bucket_capacity = None if lsh_bucket_capacity is None else int(lsh_bucket_capacity)
        self.lsh_num_hashes = None if lsh_num_hashes is None else int(lsh_num_hashes)

        self._cache = self._create_cache()

    def set_dim(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = int(dim)
        if self.policy.startswith("lsh_"):
            self._cache = self._create_cache()

    def _create_cache(self):
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not (0.0 < self.tolerance <= 1.0):
            raise ValueError("tolerance must be in (0, 1]")

        if self.policy == "fifo":
            return proximipy.FifoCache(self.capacity)

        if self.policy == "lru":
            return proximipy.LruCache(self.capacity)

        # LSH policies
        if self.dim is None or self.dim <= 0:
            raise ValueError("dim must be provided and > 0 for LSH cache policies")
        if self.lsh_bucket_capacity is None or self.lsh_bucket_capacity <= 0:
            raise ValueError("lsh_bucket_capacity must be > 0 for LSH cache policies")
        if self.lsh_num_hashes is None or self.lsh_num_hashes <= 0:
            raise ValueError("lsh_num_hashes must be > 0 for LSH cache policies")

        cls = proximipy.LshFifoCache if self.policy == "lsh_fifo" else proximipy.LshLruCache
        return cls(
            int(self.lsh_num_hashes),
            int(self.dim),
            int(self.lsh_bucket_capacity),
            None if self.seed is None else int(self.seed),
        )

    @staticmethod
    def _as_key(vec: Any) -> list[float]:
        if isinstance(vec, np.ndarray):
            return vec.astype(np.float32).tolist()
        return list(vec)

    def insert(self, key: Any, value: Any) -> None:
        self._cache.insert(self._as_key(key), value, tolerance=self.tolerance)

    def find(self, key: Any) -> Any:
        return self._cache.find(self._as_key(key))

    def find_many(self, vecs: Any) -> list[Any]:
        return [self.find(v) for v in vecs]

    def insert_many(self, vecs: Any, values: Sequence[Any]) -> None:
        for v, val in zip(vecs, values):
            self.insert(v, val)

    def cached_search(
        self,
        vecs: np.ndarray,
        *,
        k: int,
        backend_index: Any,
        cache_key_k: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(vecs, np.ndarray):
            raise TypeError("vecs must be a numpy ndarray")
        if vecs.ndim != 2:
            raise ValueError("vecs must have shape (B, D)")
        if k <= 0:
            raise ValueError("k must be > 0")

        cache_res = self.find_many(vecs)

        hit_idxs = [i for i, r in enumerate(cache_res) if r is not None]
        miss_idxs = [i for i, r in enumerate(cache_res) if r is None]

        self.cache_hit_count += len(hit_idxs)
        self.cache_miss_count += len(miss_idxs)

        B = vecs.shape[0]
        distances: list[Optional[np.ndarray]] = [None] * B
        indices: list[Optional[np.ndarray]] = [None] * B

        # Fill hits; if k mismatches, convert to miss (avoid duplicate miss indices)
        extra_misses: list[int] = []
        for i in hit_idxs:
            cached = cache_res[i]
            if isinstance(cached, dict):
                if (not cache_key_k) or (cached.get("k") == k):
                    distances[i] = np.asarray(cached["distances"], dtype=np.float32)
                    indices[i] = np.asarray(cached["indices"])
                else:
                    extra_misses.append(i)
            else:
                indices[i] = np.asarray(cached)
                distances[i] = np.zeros((k,), dtype=np.float32)

        if extra_misses:
            miss_set = set(miss_idxs)
            miss_set.update(extra_misses)
            miss_idxs = sorted(miss_set)

        # Backend search for misses
        if miss_idxs:
            missed = vecs[miss_idxs]
            d_miss, i_miss = backend_index.search(missed, k)

            for j, orig_i in enumerate(miss_idxs):
                distances[orig_i] = np.asarray(d_miss[j], dtype=np.float32)
                indices[orig_i] = np.asarray(i_miss[j])

                self.insert(
                    vecs[orig_i],
                    {
                        "k": k,
                        "distances": distances[orig_i].tolist(),
                        "indices": indices[orig_i].tolist(),
                    },
                )

        # Safety: ensure everything filled
        if any(d is None for d in distances) or any(ix is None for ix in indices):
            raise RuntimeError("cached_search: some rows were not filled (bug in cache merge logic)")

        return np.vstack(distances), np.vstack(indices)

    def get_stats(self) -> dict[str, Any]:
        denom = self.cache_hit_count + self.cache_miss_count
        return {
            "cache_name": "proximity",
            "policy": self.policy,
            "capacity": self.capacity,
            "tolerance": self.tolerance,
            "lsh_num_hashes": self.lsh_num_hashes,
            "dim": self.dim,
            "lsh_bucket_capacity": self.lsh_bucket_capacity,
            "seed": self.seed,
            "hit_count": self.cache_hit_count,
            "miss_count": self.cache_miss_count,
            "hit_rate": (self.cache_hit_count / denom) if denom > 0 else 0.0,
        }


# -------------------------
# Builder
# -------------------------

def build_cache(cfg: Optional[CacheConfig], dim: Optional[int] = None, seed: int = None) -> Optional[ProximityCache]:
    """
    Build runtime cache from your CacheConfig wrapper.

    If you use LSH policies, pass dim (embedding dimension) here.
    """
    cache_type = cfg.type 

    if cache_type == "proximity":
        sub: ProximityCacheConfig = cfg
        return ProximityCache(
            policy=sub.policy,
            tolerance=sub.tolerance,
            capacity=sub.capacity,
            lsh_bucket_capacity=sub.lsh_bucket_capacity,
            lsh_num_hashes=sub.lsh_num_hashes,
            seed=seed,
            dim=dim,
        )
    raise ValueError(f"Unknown CacheConfig.type: {cache_type!r}")

