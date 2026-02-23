from flask import config
import torch
from pwrag.args.args import AppConfig
from transformers import AutoConfig

def get_generator(config: AppConfig):
    """Automatically select generator class based on config."""
    if config.generator.generator_framework == 'openai':
        raise NotImplementedError("OpenAI API is not supported yet.")
    if config.generator.generator_framework == "vllm":
        raise NotImplementedError("VLLM is not supported yet.")
    elif config.generator.generator_framework == "fschat":
        raise NotImplementedError("FastChat is not supported yet.")
    elif config.generator.generator_framework == "hf":
        from pwrag.generator.generator import HFCausalLMGenerator
        if "t5" in config.generator.generator_model.lower() or "bart" in config.generator.generator_model.lower():
            raise NotImplementedError("EncoderDecoderGenerator is not supported yet.")
        return HFCausalLMGenerator(config)
    else:
        raise NotImplementedError("Unsupported generator framework: {}".format(config.generator.generator_framework))

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

def get_refiner(config):
    pass

def get_judger(config):
    pass

def get_cache(config):
    pass
