# generators.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# -----------------------------
# Config
# -----------------------------
@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0


# -----------------------------
# Base interface
# -----------------------------
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


# -----------------------------
# REAL MODELS
# -----------------------------
# wrapper around a Hugging Face decoder-only language model.
class HFCausalGenerator(BaseGenerator):
    """GPT/LLaMA/Mistral-style decoder-only generator"""

    def __init__(self, model_name: str, device: str, gen_config: GenerationConfig):
        self.device = device
        self.cfg = gen_config

        #loads the tokenizer that was used to train the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        #loads the actual model
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()  # disables gradient tracking (faster + less memory for inference)
    def generate(self, prompt: str) -> str:
        # 1) Convert the text prompt into token IDs the model understands
        #    - truncation: cut off if too long
        #    - padding: add pad tokens so tensors are rectangular
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # 2) Move token tensors to the same device as the model (CPU or GPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 3) Ask the model to generate new tokens autoregressively
        #    The model will: read the prompt tokens, predict the next token, append it, repeat until EOS or max_new_tokens
        out = self.model.generate(
            **inputs,
            # Hard limit on how many *new* tokens may be generated
            max_new_tokens=self.cfg.max_new_tokens,

            # Whether to sample probabilistically or generate deterministically
            do_sample=self.cfg.do_sample,
            # Sampling controls (only used if do_sample=True)
            temperature=self.cfg.temperature if self.cfg.do_sample else None,
            top_p=self.cfg.top_p if self.cfg.do_sample else None,
            top_k=self.cfg.top_k if self.cfg.do_sample else None,

            # Token that tells the model to stop generating
            eos_token_id=self.tokenizer.eos_token_id,

            # Token used for padding when batching
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # 4) Convert generated token IDs back into readable text
        #    NOTE: this includes the original prompt + generated continuation
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------
# SYNTHETIC / SIMULATED
# -----------------------------
class EchoGenerator(BaseGenerator):
    """
    Returns the retrieved context verbatim.
    Perfect for debugging retrieval.
    """
    def generate(self, prompt: str) -> str:
        return prompt


class SyntheticAnswerGenerator(BaseGenerator):
    """
    Deterministic fake generator.
    Purpose:
      - Debug RAG without an LLM
      - Simulate latency in a controlled way
      - Verify retrieval + prompt wiring
    This version simulates a fixed generation delay.
    """
    def __init__(self, sleep_seconds: float = 0.2):
        # Total fake latency (seconds)
        self.sleep_seconds = sleep_seconds

    def generate(self, prompt: str) -> str:
        # Simulate model latency (blocking)
        time.sleep(self.sleep_seconds)

        # Extract retrieved context from the prompt
        match = re.search(r"Context:\n(.+?)\n\nAnswer:", prompt, re.S)
        context = match.group(1).strip() if match else "(no context)"
        return (
            "SYNTHETIC ANSWER\n"
            "----------------\n"
            f"(simulated latency: {self.sleep_seconds:.2f}s)\n\n"
            f"Retrieved context:\n{context}"
        )
    
# -----------------------------
# Factory
# -----------------------------
def load_generator(
    generator_name: str,
    device: str,
    gen_config: Optional[GenerationConfig] = None,
    sleep_seconds_for_synthetic: float = 0,
) -> BaseGenerator:
    cfg = gen_config or GenerationConfig()
    name = generator_name.lower()

    # Synthetic / debug generators
    if name == "echo":
        return EchoGenerator()

    if name in {"synthetic", "simulated"}:
        return SyntheticAnswerGenerator(
            sleep_seconds=sleep_seconds_for_synthetic
        )

    # Default: causal LM
    return HFCausalGenerator(generator_name, device=device, gen_config=cfg)
