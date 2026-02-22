from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.generator.generator import HFCausalLMGenerator

def _flatten_record(self, record: dict) -> dict:
    flat = {}

    for k, v in record.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v

    return flat


def get_generator(config: AppConfig):
    """Automatically select generator class based on config."""

    if config.generator.generator_framework == 'openai':
        raise NotImplementedError("OpenAI API is not supported yet.")
    if config.generator.generator_framework == "vllm":
        raise NotImplementedError("VLLM is not supported yet.")
    elif config.generator.generator_framework == "fschat":
        raise NotImplementedError("FastChat is not supported yet.")
    elif config.generator.generator_framework == "hf":
        if "t5" in config.generator.generator_model.lower() or "bart" in config.generator.generator_model.lower():
            raise NotImplementedError("EncoderDecoderGenerator is not supported yet.")
        return HFCausalLMGenerator(config)
    else:
        raise NotImplementedError("Unsupported generator framework: {}".format(config.generator.generator_framework))
 
def get_retriever(config):
    pass

def get_refiner(config):
    pass

def get_judger(config):
    pass

def get_cache(config):
    pass

