from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# -------------------------
# Index
# -------------------------
IndexType = Literal["flat", "ivf", "hnsw"]
Metric = Literal["ip", "l2"]


@dataclass
class IndexConfig:
    type: IndexType
    path: str
    index_vector_count: int
    metric: Metric = "ip"
    use_gpu: bool = False

    # IVF search-time
    nprobe: int = 16
    # HNSW search-time
    ef_search: int = 64
    

    # Optional reference name if you're selecting docstores by name in Hydra.
    # (The service still takes an actual DocStoreConfig; this is just a label/reference.)
    docstore_name: Optional[str] = None  # e.g. "wiki_dpr_100k"

    def __post_init__(self) -> None:
        if self.nprobe <= 0:
            raise ValueError("nprobe must be > 0")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be > 0")


# -------------------------
# Cache
# -------------------------
CacheType = Literal["proximity"]

@dataclass
class ProximityCacheConfig:
    policy: str
    tolerance: float
    capacity: int
    name: str = "proximity"
    # Only required for LSH policies
    lsh_bucket_capacity: Optional[int] = None
    lsh_num_hashes: Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.tolerance <= 1.0):
            raise ValueError("tolerance must be in (0,1]")
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        is_lsh = self.policy.startswith("lsh_")
        if is_lsh:
            if self.lsh_bucket_capacity is None or self.lsh_bucket_capacity <= 0:
                raise ValueError("lsh_bucket_capacity must be > 0 for lsh_* policies")
            if self.lsh_num_hashes is None or self.lsh_num_hashes <= 0:
                raise ValueError("lsh_num_hashes must be > 0 for lsh_* policies")
        else:
            # Keep the config clean: disallow LSH-only knobs for non-LSH policies
            if self.lsh_bucket_capacity is not None or self.lsh_num_hashes is not None:
                raise ValueError("LSH fields must be unset unless policy is lsh_*")


@dataclass
class CacheConfig:
    """
    Tagged-union wrapper: exactly one sub-config must be set, matching `type`.
    """
    type: CacheType
    proximity: Optional[ProximityCacheConfig] = None

    def __post_init__(self) -> None:
        mapping = {"proximity": self.proximity}
        chosen = mapping.get(self.type)
        if chosen is None:
            raise ValueError(f"CacheConfig.type={self.type!r} but matching config is None")
        



# -------------------------
# Embedders
# -------------------------
EmbedderType = Literal["dpr", "synthetic", "gemma"]


@dataclass
class DPREmbedderConfig:
    type: Literal["dpr"] = "dpr"
    passage_encoder_id: str = ""
    query_encoder_id: str = ""
    batch_size: int = 32
    device: str = "auto"

    # Explicitly include the field that was referenced before.
    # DPR typically uses dot-product geometry, so don't normalize by default.
    normalize_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self.passage_encoder_id:
            raise ValueError("passage_encoder_id must be non-empty")
        if not self.query_encoder_id:
            raise ValueError("query_encoder_id must be non-empty")
        if self.normalize_embeddings:
            raise ValueError("DPR should use normalize_embeddings=false (dot-product geometry)")


@dataclass
class SyntheticEmbedderConfig:
    type: Literal["synthetic"] = "synthetic"
    dim: int = 768
    sleep_time: float = 0.0
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.sleep_time < 0:
            raise ValueError("sleep_time must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")


@dataclass
class GemmaEmbedderConfig:
    type: Literal["gemma"] = "gemma"
    model_name: str = ""
    dim: int = 768
    name: str = "gemma_embedder"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.dim <= 0:
            raise ValueError("dim must be > 0")


@dataclass
class EmbedderConfig:
    """
    Tagged-union wrapper: exactly one sub-config must be set, matching `type`.
    """
    type: EmbedderType
    dpr: Optional[DPREmbedderConfig] = None
    synthetic: Optional[SyntheticEmbedderConfig] = None
    gemma: Optional[GemmaEmbedderConfig] = None

    def __post_init__(self) -> None:
        mapping = {"dpr": self.dpr, "synthetic": self.synthetic, "gemma": self.gemma}
        chosen = mapping.get(self.type)

        if chosen is None:
            raise ValueError(f"EmbedderConfig.type={self.type!r} but matching config is None")

        extras = [k for k, v in mapping.items() if k != self.type and v is not None]
        if extras:
            raise ValueError(f"EmbedderConfig has extra configs set: {extras}")


# -------------------------
# Generators
# -------------------------
GeneratorType = Literal["llama3_instruct", "qwen2_5_instruct", "synthetic"]

@dataclass
class Qwen2_5InstructGeneratorConfig:
    type: Literal["qwen2_5_instruct"] = "qwen2_5_instruct"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = False
    max_new_tokens: int = 256
    device: str = "auto"


    use_4bit: bool = False
    hf_token: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0,1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@dataclass
class SyntheticGeneratorConfig:
    type: Literal["synthetic"] = "synthetic"
    sleep_time: float = 0.0
    response_prefix: str = "SYNTHETIC:"

    def __post_init__(self) -> None:
        if self.sleep_time < 0:
            raise ValueError("sleep_time must be >= 0")


@dataclass
class Llama3InstructGeneratorConfig:
    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    max_new_tokens: int = 64
    use_4bit: bool = False
    hf_token: Optional[str] = None  # if None, read from env in your loader code
    type: Literal["llama3_instruct"] = "llama3_instruct"

    def __post_init__(self) -> None:
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


@dataclass
class GeneratorConfig:
    type: GeneratorType
    llama3_instruct: Optional[Llama3InstructGeneratorConfig] = None
    qwen2_5_instruct: Optional[Qwen2_5InstructGeneratorConfig] = None
    synthetic: Optional[SyntheticGeneratorConfig] = None

    def __post_init__(self) -> None:
        mapping = {
            "llama3_instruct": self.llama3_instruct,
            "qwen2_5_instruct": self.qwen2_5_instruct,
            "synthetic": self.synthetic,
        }

        chosen = mapping.get(self.type)
        if chosen is None:
            raise ValueError(f"GeneratorConfig.type={self.type!r} but matching config is None")

        extras = [k for k, v in mapping.items() if k != self.type and v is not None]
        if extras:
            raise ValueError(f"GeneratorConfig has extra configs set: {extras}")


# -------------------------
# DocStore
# -------------------------
DocSourceFormat = Literal["jsonl"]
DocBackendKind = Literal["local_sqlite", "local_jsonl_offsets", "s3_jsonl_offsets", "memory_jsonl"]


@dataclass
class DocStoreConfig:
    name: str
    source_uri: str  # local path or s3://...
    source_format: DocSourceFormat = "jsonl"
    source_id_key: str = "pid"
    source_title_key: str = "title"
    source_text_key: str = "text"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.source_uri:
            raise ValueError("source_uri must be non-empty")
# -------------------------
# Prompting
# -------------------------
PromptType = Literal["no_retrieval", "with_context"]


@dataclass
class TelemetryConfig:
    enabled: bool = False
    interval_s: float = 5.0
    dir: str = "logs"



# -------------------------
# Service config
# -------------------------
@dataclass
class RagServiceConfig:
    generator: GeneratorConfig
    embedder: EmbedderConfig
    index: IndexConfig
    docstore: DocStoreConfig

    artifact_dir: str
    retrieve_only: bool = False

    max_inflight: int = 64
    num_workers: int = 1

    host: str = "localhost"
    port: int = 50051
    log_level: str = "INFO"

    prompt_type: PromptType = "with_context"
    max_ctx_chars: int = 4000
    telemetry: Optional[TelemetryConfig] = None

    top_k: int = 5
    cache: Optional[CacheConfig] = None
    seed: Optional[int] = None
    docstore_backend: DocBackendKind = "memory_jsonl"

    def __post_init__(self) -> None:
        if not self.artifact_dir:
            raise ValueError("artifact_dir must be non-empty")
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be > 0")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be in [1, 65535]")


@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float
