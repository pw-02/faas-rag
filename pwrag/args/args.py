from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---- Generator ----
@dataclass
class GeneratorConfig:
    name: str
    generator_framework: str  # 'hf','vllm','fschat'
    generator_model: str = "llama3-8B-instruct"
    generator_max_input_length: int = 2048
    generator_batch_size: int = 2
    generation_params: Optional[Dict[str, Any]] = None


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
class RetrieverEmbedderConfig:
    retrieval_method: str = "dpr"
    retrieval_model: str = ""
    dim: int = 768
    metric: str = "cosine"              # cosine | ip | l2
    pooling_method: str = "cls"         # cls | mean
    normalize_embeddings: bool = True
    use_sentence_transformer: bool = True
    # device: str = "cpu:0"
    use_fp16: bool = True


@dataclass
class RetrieverIndexConfig:
    index_path: str = ""
    index_type: str = "ivf"             # flat | ivf | hnsw
    metric: str = "cosine"
    nprobe: int = 16
    ef_search: int = 64
    use_faiss_gpu: bool = False
    # faiss_index_params: Dict[str, Any] = None

@dataclass
class RetrieverSearchConfig:
    retrieval_topk: int = 5
    batch_size: int = 64
    query_max_length: int = 64

@dataclass
class RetrieverPipelineConfig:
    type: str = "dense"
    use_cache: bool = False
    use_reranker: bool = False
    reranker_model: Optional[str] = None
    reranker_topk: int = 50

@dataclass
class RetrieverConfig:
    name: str = "dense"

    # these map 1:1 to the yaml groups
    pipeline: RetrieverPipelineConfig = field(default_factory=RetrieverPipelineConfig)
    corpus: RetrieverCorpusConfig = field(default_factory=RetrieverCorpusConfig)
    embedder: RetrieverEmbedderConfig = field(default_factory=RetrieverEmbedderConfig)
    index: RetrieverIndexConfig = field(default_factory=RetrieverIndexConfig)
    search: RetrieverSearchConfig = field(default_factory=RetrieverSearchConfig)
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
    retriever_embedder_device: str = "cuda:0"
    
    metrics: List[str] = field(default_factory=lambda: ["em", "f1", "acc", "precision", "recall", "input_tokens"])
    save_sample_metrics: bool = False
    save_summary_metrics: bool = False
    is_reasoning: bool = False
    run_name: Optional[str] = None