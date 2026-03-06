from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---- Judger ----
@dataclass
class JudgerConfig:
    name: str = "adaptive"  # options: skr, adaptive, null
    model_path: Optional[str] = None
    batch_size: int = 64
    max_length: int = 128
    device: str = "cuda:0"

# ---- Generator ----
@dataclass
class GeneratorConfig:
    name: str
    framework: str  # 'hf','vllm','fschat'
    model_name: str = "llama3-8B-instruct"
    model_path: Optional[str] = None
    max_input_length: int = 2048
    batch_size: int = 2
    generation_params: Optional[Dict[str, Any]] = None
    openai_endpoint: Optional[str] = None

# ---- cache ----
@dataclass
class ProximityCacheConfig:
    type: str = "proximity"  # options: proximity, exact, none
    policy: str = "fifo" # options: fifo, lru, lsh_fifo, lsh_lru
    tolerance: float = 0.8
    capacity: int = 10
    lsh_bucket_capacity: int = 5
    lsh_num_hashes: int = 128

@dataclass
class NoneCacheConfig:
    type: str = "none"

@dataclass
class RetrieverCacheConfig:
    type: str = "proximity"  # options: proximity, exact, none
    proximity: ProximityCacheConfig = field(default_factory=ProximityCacheConfig)
    none: NoneCacheConfig = field(default_factory=NoneCacheConfig)


# ---- Retriever sub-configs ----
@dataclass
class RetrieverCorpusConfig:
    name: str = "wiki_100k"
    corpus_path: str = ""

@dataclass
class RetrieverEncoderConfig:
    retrieval_method: str = "dpr"
    retrieval_model: str = ""
    dim: int = 768
    metric: str = "cosine"              # cosine | ip | l2
    pooling_method: str = "cls"         # cls | mean
    normalize_embeddings: bool = True
    use_sentence_transformer: bool = True
    # device: str = "cpu:0"
    use_fp16: bool = True
    query_max_length: int = 512
    batch_size: int = 16


@dataclass
class RetrieverIndexConfig:
    name: str = ""
    index_path: str = ""
    index_type: str = "ivf"             # flat | ivf | hnsw
    metric: str = "cosine"
    nprobe: int = 16
    ef_search: int = 64
    use_faiss_gpu: bool = False
    # faiss_index_params: Dict[str, Any] = None

@dataclass
class RetrieverPipelineConfig:
    name: str
    use_cache: bool = False
    use_reranker: bool = False
    reranker_model: Optional[str] = None
    reranker_topk: int = 50

@dataclass
class RetrieverConfig:
    # these map 1:1 to the yaml groups
    pipeline: RetrieverPipelineConfig = field(default_factory=RetrieverPipelineConfig)
    corpus: RetrieverCorpusConfig = field(default_factory=RetrieverCorpusConfig)
    encoder: RetrieverEncoderConfig = field(default_factory=RetrieverEncoderConfig)
    index: RetrieverIndexConfig = field(default_factory=RetrieverIndexConfig)
    cache: RetrieverCacheConfig = field(default_factory=RetrieverCacheConfig)




# ---- Dataset ----
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_path: str

# ---- App ----
@dataclass
class AppConfig:
    generator: GeneratorConfig
    retriever: RetrieverConfig
    dataset: DatasetConfig
    seed: Optional[int] = None
    save_dir: str = "results"
    max_sample_num: Optional[int] = None
    random_sample: bool = False
    generator_device: str = "cuda:0"
    retriever_encoder_device: str = "cuda:0"
    metrics: List[str] = field(default_factory=lambda: ["em", "f1", "acc", "precision", "recall", "input_tokens"])
    is_reasoning: bool = False
    batch_size: int = 1
    retrieval_topk: int = 5
