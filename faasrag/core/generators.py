# faasrag/core/builders.py
from faasrag.core.args import (LlamaGeneratorConfig,)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        max_new_tokens: int,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device.startswith("cuda") else None,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample else None,
            top_k=self.top_k if self.do_sample else None,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


def build_generator(cfg: LlamaGeneratorConfig, *, device: str):
    """
    Create a runtime generator instance from LlamaGeneratorConfig.
    """
    return LlamaGenerator(
        model_name=cfg.model_name,
        device=device,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        do_sample=cfg.do_sample,
        max_new_tokens=cfg.max_new_tokens,
    )

if __name__ == "__main__":
    # Example usage
    gen_cfg = LlamaGeneratorConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        max_new_tokens=64,
    )
    generator = build_generator(gen_cfg, device="cuda")
    prompt = "What is the capital of France?"
    response = generator.generate(prompt)
    print(response)