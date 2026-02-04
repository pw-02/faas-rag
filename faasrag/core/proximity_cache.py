from __future__ import annotations

from typing import Optional, Any, Callable, Sequence
import numpy as np
import proximipy


class ProximityCache:
    def __init__(
        self,
        *,
        cache_policy: Optional[str] = None,
        tolerance: float = 0.7,
        cache_size: int = 10,
        lsh_num_hash: int = 128,
        # dim: Optional[int] = None,  # ✅ pass embedding dim explicitly
        lsh_bucket_capacity: int = 10,
        seed: Optional[int] = 42,
    ):
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        self.tolerance = float(tolerance)
        #ensure cache policy is valid
        self.cache_policy = cache_policy
        self.cache_size = int(cache_size)
        self.lsh_num_hash = int(lsh_num_hash)
        self.dim = None  # to be set later if needed
        self.lsh_bucket_capacity = int(lsh_bucket_capacity)
        self.seed = seed
        self.proximity_cache = self._create_proximity_cache()

    def _create_proximity_cache(self):

        policy = str(self.cache_policy).lower()

        #check policy is one of the supported ones
        if policy not in {"fifo", "lru", "lsh_fifo", "lsh_lru"}:
            raise ValueError(f"Unknown cache_policy={policy!r}")

        if policy == "fifo":
            return proximipy.FifoCache(self.cache_size)

        if policy == "lru":
            return proximipy.LruCache(self.cache_size)

        if policy in {"lsh_fifo", "lsh_lru"}:
            if self.dim is None or self.dim <= 0:
                raise ValueError("dim must be provided and > 0 for LSH cache policies")
            cls = proximipy.LshFifoCache if policy == "lsh_fifo" else proximipy.LshLruCache
            return cls(
                self.lsh_num_hash,
                self.dim,
                self.lsh_bucket_capacity,
                None if self.seed is None else int(self.seed),
            )

        raise ValueError(f"Unknown cache_policy={policy!r}")


    def insert(self, key: Any, value: Any) -> None:
        # store value as-is; proximipy wants "list" of values, but your wrapper can enforce it
        self.proximity_cache.insert(key, value, tolerance=self.tolerance)

    def find(self, key: Any):
        return self.proximity_cache.find(key)
    
    def find_many(self, vecs: Any) -> list[Any]:
        """
        Batch cache lookup. Returns list aligned with vecs:
        - cached value or None
        """
        out: list[Any] = []
        for v in vecs:
            out.append(self.find(list(v)))
        return out

    def insert_many(self, vecs: Any, values: Sequence[Any]) -> None:
        for v, val in zip(vecs, values):
            self.insert(list(v), val)

    def cached_search(
        self,
        vecs: np.ndarray,
        *,
        k: int,
        backend_index,
        cache_key_k: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Read-through cache around a backend search (FAISS).
        - vecs: (B, D) float32 np.ndarray
        - backend_search: function(vecs_subset, k) -> (distances, indices)

        Cache stores per-vector result. Optionally include k in cached value.
        """

        # 1) cache lookup
        cache_res = self.find_many(vecs)

        hit_idxs = [i for i, r in enumerate(cache_res) if r is not None]
        miss_idxs = [i for i, r in enumerate(cache_res) if r is None]

        self.cache_hit_count += len(hit_idxs)
        self.cache_miss_count += len(miss_idxs)

        B = vecs.shape[0]
        distances = [None] * B
        indices = [None] * B

        # 2) fill hits
        for i in hit_idxs:
            cached = cache_res[i]
            # you can choose your cached format; here's a robust one:
            # cached = {"k": k, "distances": ..., "indices": ...}
            if isinstance(cached, dict):
                if (not cache_key_k) or (cached.get("k") == k):
                    distances[i] = np.asarray(cached["distances"])
                    indices[i] = np.asarray(cached["indices"])
                else:
                    # cached for different k -> treat as miss
                    distances[i] = None
                    indices[i] = None
                    miss_idxs.append(i)
            else:
                # if you cached just indices, adapt here
                # (not recommended unless you don't care about distances)
                indices[i] = np.asarray(cached)
                distances[i] = np.zeros((k,), dtype=np.float32)

        # 3) backend for misses
        if miss_idxs:
            missed = vecs[miss_idxs]
            d_miss, i_miss = backend_index.search(missed, k)

            # 4) merge + insert
            for j, orig_i in enumerate(miss_idxs):
                distances[orig_i] = d_miss[j]
                indices[orig_i] = i_miss[j]
                self.insert(
                    list(vecs[orig_i]),
                    {"k": k, "distances": d_miss[j].tolist(), "indices": i_miss[j].tolist()},
                )

        return np.vstack(distances), np.vstack(indices)

    def get_stats(self):
        return {
            "cache_name": "proximity",
            "cache_policy": self.cache_policy,
            "cache_size": self.cache_size,
            "cache_lsh_num_hash": self.lsh_num_hash,
            "cache_dim": self.dim,
            "cache_lsh_bucket_capacity": self.lsh_bucket_capacity,
            "cache_seed": self.seed,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "cache_hit_rate": (
                self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count)
                if (self.cache_hit_count + self.cache_miss_count) > 0
                else 0.0
            ),
        }
