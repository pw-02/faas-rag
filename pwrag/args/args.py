from dataclasses import dataclass
from typing import List

@dataclass
class GeneratorConfig:
    name: str
    generator_framework: str #inference frame work of LLM, supporting: 'hf','vllm','fschat'
    generator_model: str = "llama3-8B-instruct" # name or path of the generator model
    generator_max_input_length: int = 2048  # max length of the input
    generator_batch_size: int = 2 # batch size for generation, invalid for vllm
    generation_params: dict = None # additional generation parameters, e.g., temperature, top_p, etc.

@dataclass
class RetrieverConfig:
    name: str
    index_path: str
    corpus_path: str
    retrieval_model: str
    retrieval_device: str = "cuda:0"

    index_type: str = "ivf"      # flat | ivf | hnsw
    metric: str = "cosine"      # cosine | ip | l2
    dim: int = 768

    nprobe: int = 16
    ef_search: int = 64

    normalize_embeddings: bool = True
    faiss_gpu: bool = False
    top_k: int = 5

@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_path: str



class AppConfig:
    generator: GeneratorConfig
    retriever: RetrieverConfig
    dataset: DatasetConfig
    generator_framework: str = "hf" # inference frame work of LLM, supporting: 'hf','vllm','fschat'
    generator_device: str = "cuda:0"
    retriever_device: str = "cuda:0"
    seed: int = None
    save_dir: str = "results"
    max_sample_num: int = None
    random_sample: bool = False
    metrics: List[str] = ["em", "f1", "acc", "precision", "recall", "input_tokens"]
    save_dir:str = "results"
    save_sample_metrics: bool = True
    save_summary_metrics: bool = True
    is_reasoning: bool = False




