from pwrag.args.args import AppConfig
from transformers import AutoConfig
from contextlib import contextmanager
import time
from typing import Any

@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    if key not in store or not isinstance(store[key], (int, float)):
        store[key] = time.perf_counter() - t0
    else:
        store[key] += time.perf_counter() - t0













class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        try:
            v = float(val)
        except Exception:
            return
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


def get_generator(config: AppConfig):
    """Automatically select generator class based on config."""
    if config.generator.framework == 'openai':
        raise NotImplementedError("OpenAI API is not supported yet.")
    if config.generator.framework == "vllm":
        from pwrag.generator.generator import VLLMGenerator
        return VLLMGenerator(config)
    elif config.generator.framework == "fschat":
        raise NotImplementedError("FastChat is not supported yet.")
    elif config.generator.framework == "hf":
        from pwrag.generator.generator import HFCausalLMGenerator
        if "t5" in config.generator.model_name.lower() or "bart" in config.generator.model_name.lower():
            raise NotImplementedError("EncoderDecoderGenerator is not supported yet.")
        return HFCausalLMGenerator(config)
    else:
        raise NotImplementedError("Unsupported generator framework: {}".format(config.generator.framework))

def get_retriever(config: AppConfig):

    from pwrag.retriever.retriever import DenseRetriever
    return DenseRetriever(config) 


    # # if config.use_multi_retriever:
    # #     # must load special class for manage multi retriever
    # #     from pwrag.retriever.retriever import MultiRetrieverRouter
    # #     return MultiRetrieverRouter(config)
    
    # if config.retriever.retrieval_model == "bm25":
    #     from pwrag.retriever.retriever import BM25Retriever
    #     return BM25Retriever(config)
    # elif config.retriever.retrieval_model == "splade":
    #     raise NotImplementedError("SPLADE retriever is not supported yet.")
    #     # from pwrag.retriever.retriever import SPLADERetriever
    #     # return SPLADERetriever(config)
    # else:
        # try:
        #     model_config = AutoConfig.from_pretrained(config.retriever.retrieval_model_path)
        #     arch = model_config.architectures[0]
        #     if "clip" in arch.lower():
        #         from pwrag.retriever.retriever import MultiModalRetriever
        #         return MultiModalRetriever(config)
        #         # return getattr(importlib.import_module("flashrag.retriever"), "MultiModalRetriever")(config)
        #     else:
        #         from pwrag.retriever.retriever import DenseRetriever
        #         return DenseRetriever(config)
        #         # return getattr(importlib.import_module("flashrag.retriever"), "DenseRetriever")(config)
        # except:
        #         from pwrag.retriever.retriever import DenseRetriever
        #         return DenseRetriever(config)


def get_reranker(config):
    # model_path = config["rerank_model_path"]
    # # get model config
    # model_config = AutoConfig.from_pretrained(model_path)
    # arch = model_config.architectures[0]
    # if "forsequenceclassification" in arch.lower():
    #     return getattr(importlib.import_module("flashrag.retriever"), "CrossReranker")(config)
    # else:
    #     return getattr(importlib.import_module("flashrag.retriever"), "BiReranker")(config)
    raise NotImplementedError("Reranker is not supported yet.")


def get_cache(config: AppConfig):    # if config.retriever.pipeline.use_cache:
    #     return getattr(importlib.import_module("flashrag.retriever"), "CacheManager")(config)
    if config.retriever.cache.type == "proximity":
        from pwrag.retriever.caches import ProximityCache
        return ProximityCache(
            policy=config.retriever.cache.proximity.policy,
            tolerance=config.retriever.cache.proximity.tolerance,
            capacity=config.retriever.cache.proximity.capacity,
            lsh_bucket_capacity=config.retriever.cache.proximity.lsh_bucket_capacity,
            lsh_num_hashes=config.retriever.cache.proximity.lsh_num_hashes,
            lsh_dim=config.retriever.embedder.dim,
            lsh_seed=config.seed
        )
    
    raise NotImplementedError("Cache is not supported yet.")

def get_refiner(config):
    pass

def get_judger(config:AppConfig):
    judger_name = config.judger_name
    if "skr" in judger_name.lower():
        from pwrag.judger.judger import SKRJudger
        return SKRJudger(config)
    elif "adaptive" in judger_name.lower():
        from pwrag.judger.judger import AdaptiveJudger
        return AdaptiveJudger(config)
    else:
        assert False, "No implementation!"

