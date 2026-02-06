from __future__ import annotations

import time
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faasrag.core.args import GeneratorConfig, Llama3InstructGeneratorConfig, Qwen2_5InstructGeneratorConfig, SyntheticGeneratorConfig
from faasrag.core.prompts import extract_short_answer


# -------------------------
# Runtime generators
# -------------------------
class SyntheticGenerator:
    def __init__(self, sleep_seconds: float, response_prefix: str):
        self.sleep_seconds = float(sleep_seconds)
        self.response_prefix = response_prefix

    def generate(self, prompt: str) -> tuple[str, int]:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        text = f"{self.response_prefix} {prompt}"
        return text, 0


class HFCausalLMGenerator:
    """
    Generic HF CausalLM generator (LLaMA, Qwen, Mistral, etc.)
    via AutoModelForCausalLM + AutoTokenizer.

    Exposes:
      - generate(prompt: str)
      - generate_messages(messages: list[dict[str,str]])  (chat template if available)
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        max_new_tokens: int,
        use_4bit: bool = False,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.do_sample = bool(do_sample)
        self.max_new_tokens = int(max_new_tokens)
        self.use_4bit = bool(use_4bit)
        self.hf_token = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {}

        # dtype / device mapping
        if device.startswith("cuda"):
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32

        # Optional 4-bit quantization (typically CUDA-only)
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                kwargs["device_map"] = "auto"
            except Exception as e:
                raise RuntimeError(
                    "use_4bit=True requires bitsandbytes + compatible environment. "
                    "Install with `pip install bitsandbytes` (CUDA setups)."
                ) from e

        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, **kwargs)
        self.model.eval()

    def _move_inputs_to_model_device(self, inputs: dict) -> dict:
        # Works even when using device_map="auto"
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, prompt: str) -> tuple[str, int]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = self._move_inputs_to_model_device(inputs)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # text=extract_short_answer(text)
        return text, int(gen_ids.shape[-1])

    @torch.no_grad()
    def generate_messages(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        """
        Chat-style generation using tokenizer chat template when available.

        messages example:
          [{"role":"system","content":"..."},{"role":"user","content":"..."}]
        """
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: simple concat (not as good as true templates)
            prompt = "\n".join(f'{m["role"]}: {m["content"]}' for m in messages) + "\nassistant:"

        return self.generate(prompt)


# -------------------------
# Builder
# -------------------------
Generator = Union[HFCausalLMGenerator, SyntheticGenerator]


def build_generator(cfg: GeneratorConfig, device: str) -> Generator:
    """
    Build a generator from the tagged-union GeneratorConfig wrapper.

    Expected wrapper shape:
      cfg.type in {"llama3_instruct", "qwen2_5_instruct", "synthetic"}
      and the matching sub-config is set (cfg.llama3_instruct / cfg.qwen2_5_instruct / cfg.synthetic).
    """
    if cfg.type == "synthetic":
        sub: SyntheticGeneratorConfig = cfg
        return SyntheticGenerator(
            sleep_seconds=sub.sleep_time,
            response_prefix=sub.response_prefix,
        )
    if cfg.type == "llama3_instruct":
        sub: Llama3InstructGeneratorConfig = cfg
    elif cfg.type == "qwen2_5_instruct":
        sub: Qwen2_5InstructGeneratorConfig = cfg
    else:
        raise ValueError(f"Unknown GeneratorConfig.type: {cfg.type!r}")

    # LLaMA + Qwen both use the same runtime HF generator
    return HFCausalLMGenerator(
        model_name=sub.model_name,
        device=device,
        temperature=sub.temperature,
        top_p=sub.top_p,
        top_k=sub.top_k,
        do_sample=sub.do_sample,
        max_new_tokens=sub.max_new_tokens,
        use_4bit=sub.use_4bit,
        hf_token=sub.hf_token,
    )
