from typing import Optional
import proximipy
from core.rag_base import RagPipelineBase

class RagPipelineProximity(RagPipelineBase):
    def __init__(
        self,
        *,
        cache_policy: Optional[str] = None,
        cache_size: int = 10,
        lsh_cache_num_hash: int = 128,
        lsh_cache_expected_dim: int = 0,
        lsh_cache_bucket_capacity: int = 10,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        self.cache_policy = cache_policy
        self.cache_size = cache_size
        self.lsh_cache_num_hash = lsh_cache_num_hash
        self.lsh_cache_expected_dim = lsh_cache_expected_dim
        self.lsh_cache_bucket_capacity = lsh_cache_bucket_capacity
        self.seed = seed

        super().__init__(**kwargs)

        # now that self.index exists (created in base __init__), build cache
        self.proximity_cache = self.create_proxixmity_cache()

    def create_proxixmity_cache(self):
        policy = self.cache_policy
        if policy is None:
            return None
        policy = str(policy).lower()

        if policy == "fifo":
            return proximipy.FifoCache(int(self.cache_size))

        if policy == "lru":
            return proximipy.LruCache(int(self.cache_size))

        if policy == "lsh_fifo":
            dim = int(self.lsh_cache_expected_dim) or int(self.index.d)
            return proximipy.LshFifoCache(
                int(self.lsh_cache_num_hash),
                dim,
                int(self.lsh_cache_bucket_capacity),
                None if self.seed is None else int(self.seed),
            )

        if policy == "lsh_lru":
            dim = int(self.lsh_cache_expected_dim) or int(self.index.d)
            return proximipy.LshLruCache(
                int(self.lsh_cache_num_hash),
                dim,
                int(self.lsh_cache_bucket_capacity),
                None if self.seed is None else int(self.seed),
            )

        raise ValueError(f"Unknown cache_policy={policy!r}")
    

    
    