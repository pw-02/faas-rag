from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, List

import numpy as np
import proximipy

T = TypeVar("T")

class ProximityCache:

    def __init__(
        self,
        policy: str,
        tolerance: float,
        capacity: int,
        # LSH-only:
        lsh_bucket_capacity: Optional[int] = None,
        lsh_num_hashes: Optional[int] = None,
        lsh_seed: Optional[int] = None,
        lsh_dim: Optional[int] = None,
        # If True, enforce dim check for non-LSH too (optional)
        strict_dim: bool = False,
    ) -> None:
        self.policy = self._normalize_policy(policy)
        self.tolerance = float(tolerance)
        self.capacity = int(capacity)

        self.lsh_dim = lsh_dim
        self.strict_dim = strict_dim
        self.lsh_num_hashes = lsh_num_hashes
        self.lsh_bucket_capacity = lsh_bucket_capacity
        self.lsh_seed = lsh_seed

        if self.policy not in {"fifo", "lru", "lsh_fifo", "lsh_lru"}:
            raise ValueError(f"Unknown policy={policy!r} (normalized={self.policy!r})")
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not (0.0 < self.tolerance <= 1.0):
            raise ValueError("tolerance must be in (0, 1]")
        self._cache = self._create_cache()

    def _create_cache(self):
        if self.policy == "fifo":
            return proximipy.FifoCache(self.capacity)
        if self.policy == "lru":
            return proximipy.LruCache(self.capacity)

        # LSH policies
        if self.policy == "lsh_fifo":
            return proximipy.LshFifoCache(
                self.lsh_num_hashes,
                self.lsh_dim,
                self.lsh_bucket_capacity,
                self.lsh_seed,
            )
        else:
            return proximipy.LshLruCache(
                self.lsh_num_hashes,
                self.lsh_dim,
                self.lsh_bucket_capacity,
                self.lsh_seed,
            )


    # -------------------------
    # Helpers (inside the class)
    # -------------------------

    def _normalize_policy(self, policy: str) -> str:
        p = str(policy).lower()
        return {"lpt": "lru", "lsh_lpt": "lsh_lru"}.get(p, p)

    def _key_dim_for_validation(self) -> Optional[int]:
        # For LSH, always enforce dim
        if self.policy.startswith("lsh_"):
            return self.lsh_dim
        # For non-LSH, only enforce if strict_dim=True
        return self.lsh_dim if self.strict_dim else None

    def _to_key_list(self, key: Any, *, dim: Optional[int]) -> List[float]:
        """
        Convert user key into list[float] expected by proximipy.
        Accepts:
          - list/tuple
          - 1D numpy array
          - other iterables
        """
        if isinstance(key, np.ndarray):
            if key.ndim != 1:
                raise ValueError(f"Key ndarray must be 1D, got shape={key.shape}")
            out = key.astype(np.float32, copy=False).tolist()
        elif isinstance(key, (list, tuple)):
            out = list(key)
        else:
            try:
                out = list(key)  # type: ignore[arg-type]
            except TypeError as e:
                raise TypeError(
                    f"Key type {type(key)!r} is not supported. "
                    "Use a 1D numpy array or a sequence of floats."
                ) from e

        if dim is not None and len(out) != dim:
            raise ValueError(f"Key dim mismatch: expected {dim}, got {len(out)}")

        try:
            return [float(x) for x in out]
        except Exception as e:
            raise TypeError("Key contains non-numeric values") from e

    def _to_key_batch(self, keys: Iterable[Any], *, dim: Optional[int]) -> List[List[float]]:
        return [self._to_key_list(k, dim=dim) for k in keys]
    
    # -------------------------
    # Public API
    # -------------------------

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return self.find(key) is not None

    def insert(self, key: Any, value: Any) -> None:
        dim = self._key_dim_for_validation()
        k = self._to_key_list(key, dim=dim)
        self._cache.insert(k, value, tolerance=self.tolerance)

    def find(self, key: Any) -> Optional[Any]:
        dim = self._key_dim_for_validation()
        k = self._to_key_list(key, dim=dim)
        return self._cache.find(k)

    def batch_find(self, keys: Sequence[Any]) -> List[Optional[Any]]:
        dim = self._key_dim_for_validation()
        ks = self._to_key_batch(keys, dim=dim)
        return self._cache.batch_find(ks)

    def get_or_compute(self, key: Any, compute: Callable[[Any], T]) -> T:
        hit = self.find(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        val = compute(key)
        self.insert(key, val)
        return val

    def clear(self) -> None:
        clear_fn = getattr(self._cache, "clear", None)
        if clear_fn is None:
            raise AttributeError("Underlying cache does not expose clear()")
        clear_fn()