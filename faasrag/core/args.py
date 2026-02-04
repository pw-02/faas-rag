from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Union

# -------------------------
# Cache
# -------------------------
CachePolicy = Literal["fifo", "lpt", "lsh_fifo", "lsh_lpt"]

@dataclass
class ProximityCacheConfig:
    policy: CachePolicy
    tolerance: float
    capacity: int
    lsh_bucket_capacity: int
    lsh_num_hashes: int
    name: str = "proximity"

    def __post_init__(self):
        if not (0.0 < self.tolerance <= 1.0):
            raise ValueError("tolerance must be in (0,1]")
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.lsh_bucket_capacity <= 0:
            raise ValueError("lsh_bucket_capacity must be > 0")
        if self.lsh_num_hashes <= 0:
            raise ValueError("lsh_num_hashes must be > 0")


# -------------------------
# Embedders
# -------------------------
Metric = Literal["ip", "l2"]

@dataclass
class DPRNQEmbedderConfig:
    passage_encoder: str
    question_encoder: str
    metric: Metric
    normalize: bool
    dim: int
    name: str = "dpr_nq_embedder"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.normalize:
            raise ValueError("DPR should use normalize=false (dot-product geometry)")


@dataclass
class DPRMultiSetEmbedderConfig:
    passage_encoder: str
    question_encoder: str
    metric: Metric
    normalize: bool
    dim: int
    name: str = "dpr_multiset_embedder"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.normalize:
            raise ValueError("DPR should use normalize=false (dot-product geometry)")


@dataclass
class SyntheticEmbedderConfig:
    dim: int
    sleep_time: float
    query_prefix: str
    passage_prefix: str
    name: str = "synthetic_embedder"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.sleep_time < 0:
            raise ValueError("sleep_time must be >= 0")


@dataclass
class GemmaEmbedderConfig:
    model_name: str
    dim: int
    name: str = "gemma_embedder"

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be > 0")


# If you want one wrapper config, do it as a tagged union.
EmbedderType = Literal["dpr_nq", "dpr_multiset", "synthetic", "gemma"]

@dataclass
class EmbedderConfig:
    type: EmbedderType
    dpr_nq: Optional[DPRNQEmbedderConfig] = None
    dpr_multiset: Optional[DPRMultiSetEmbedderConfig] = None
    synthetic: Optional[SyntheticEmbedderConfig] = None
    gemma: Optional[GemmaEmbedderConfig] = None

    def __post_init__(self):
        # Enforce exactly one sub-config present and matching `type`
        mapping = {
            "dpr_nq": self.dpr_nq,
            "dpr_multiset": self.dpr_multiset,
            "synthetic": self.synthetic,
            "gemma": self.gemma,
        }
        chosen = mapping.get(self.type)

        if chosen is None:
            raise ValueError(f"EmbedderConfig.type={self.type!r} but matching config is None")

        # Ensure no extra configs are set
        extras = [k for k, v in mapping.items() if k != self.type and v is not None]
        if extras:
            raise ValueError(f"EmbedderConfig has extra configs set: {extras}")


# -------------------------
# Generator (LLaMA)
# -------------------------
@dataclass
class LlamaGeneratorConfig:
    name: str
    model_name: str
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    max_new_tokens: int

    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0,1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

        if not self.do_sample and self.temperature != 0.0:
            print("⚠️ do_sample=false: temperature is usually ignored by backends")

        if "instruct" not in self.model_name.lower():
            print("⚠️ model_name does not look like an Instruct model")


# faasrag/core/config_schema.py

@dataclass
class RagServiceConfig:
    generator: LlamaGeneratorConfig
    embedder: EmbedderConfig
    cache: Optional[ProximityCacheConfig] = None
    vector_index_path: str
    docstore_path: str
    top_k: int = 5
    device: str = "auto"
    cache: Optional[ProximityCacheConfig] = None
