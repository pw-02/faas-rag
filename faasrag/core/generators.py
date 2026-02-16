from __future__ import annotations

import time
from typing import Any, Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from faasrag.core.args import (GeneratorConfig, Llama3InstructGeneratorConfig, 
                               Qwen2_5InstructGeneratorConfig, SyntheticGeneratorConfig,
                                 vLLMOpenAIGeneratorConfig)

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import time
import openai


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: Dict[str, Any]  # ttft_s, total_s, prefill_tps, decode_tps, finish_reason


class VLLMStreamingGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout_s: float = 60.0,
    ):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
        )
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def generate(self, prompt: str) -> GenResult:
        start_time = time.time()
        first_token_time: Optional[float] = None

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        full_text = ""
        usage = None
        finish_reason = None

        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                finish_reason = getattr(choice, "finish_reason", finish_reason)

                delta = choice.delta.content
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_text += delta

            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

        end_time = time.time()

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

        ttft_s = (first_token_time - start_time) if first_token_time else None
        total_s = end_time - start_time

        # Derived rates
        prefill_tps = None
        decode_tps = None

        if ttft_s is not None and ttft_s > 0 and prompt_tokens > 0:
            prefill_tps = prompt_tokens / ttft_s

        if ttft_s is not None:
            decode_time_s = max(total_s - ttft_s, 1e-9)
            if completion_tokens > 0:
                decode_tps = completion_tokens / decode_time_s

        metrics = {
            "ttft_s": ttft_s,
            "total_s": total_s,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "finish_reason": finish_reason,
        }

        return GenResult(
            text=full_text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metrics=metrics,
        )










class VLLMOpenAIGenerator:
    """
    Calls a vLLM OpenAI-compatible server.
    Keeps the same interface as your HF generator: generate(prompt:str) -> (text, prompt_tokens, completion_tokens, total_tokens)
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout_s: float = 60.0,
        use_chat_completions: bool = True,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.use_chat_completions = bool(use_chat_completions)

    def generate(self, prompt: str) -> Tuple[str, int, int, int]:
        if self.use_chat_completions:
            # Treat your whole prompt as a single user message (works fine for string prompts)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
        else:
            # Legacy completions endpoint (some servers/models)
            resp = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = (resp.choices[0].text or "").strip()

        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

        return text, prompt_tokens, completion_tokens, total_tokens




# -------------------------
# Runtime generators
# -------------------------
class SyntheticGenerator:
    def __init__(self, sleep_seconds: float, response_prefix: str):
        self.sleep_seconds = float(sleep_seconds)
        self.response_prefix = response_prefix

    def _approx_tokens(self, s: str) -> int:
        # cheap, stable heuristic: ~4 chars/token (very rough)
        s = s or ""
        return max(1, len(s) // 4) if s else 0

    def generate(self, prompt: str) -> tuple[str, int, int, int]:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        text = f"{self.response_prefix} {prompt}"

        prompt_tokens = self._approx_tokens(prompt)
        completion_tokens = self._approx_tokens(text) - prompt_tokens
        completion_tokens = max(0, completion_tokens)
        total_tokens = prompt_tokens + completion_tokens

        return text, prompt_tokens, completion_tokens, total_tokens


class HFCausalLMGenerator:
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {}
        is_cuda = device.startswith("cuda")

        # 4-bit quantization (CUDA-only in practice)
        if self.use_4bit:
            if not is_cuda:
                raise ValueError("use_4bit=True is intended for CUDA devices only.")
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception as e:
                raise RuntimeError(
                    "use_4bit=True requires bitsandbytes. Install with `pip install bitsandbytes`."
                ) from e
        else:
            kwargs["torch_dtype"] = torch.float16 if is_cuda else torch.float32

        # Pin model to a specific device (e.g., cuda:0)
        if is_cuda:
            kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, **kwargs
        )
        self.model.eval()

    def _move_inputs_to_model_device(self, inputs: dict) -> dict:
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    @torch.no_grad()
    def generate(self, prompt: str) -> tuple[str, int, int, int]:
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

        prompt_tokens = inputs["input_ids"].shape[-1]
        total_tokens = out.shape[-1]
        completion_tokens = total_tokens - prompt_tokens
        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text, int(prompt_tokens), int(completion_tokens), int(total_tokens)


    # ---- Added for exact-length microbench ----
    @torch.no_grad()
    def generate_from_ids(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[str, int, int, int]:
        """
        Generate using exact token inputs without re-tokenizing from text.
        This is critical for an accurate token-length sweep microbenchmark.
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
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

        prompt_tokens = inputs["input_ids"].shape[-1]
        total_tokens = out.shape[-1]
        completion_tokens = total_tokens - prompt_tokens
        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text, int(prompt_tokens), int(completion_tokens), int(total_tokens)



# -------------------------
# Builder
# -------------------------
Generator = Union[HFCausalLMGenerator, SyntheticGenerator]


def build_generator(cfg: GeneratorConfig) -> Generator:
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
    elif cfg.type == "vllm":
        sub: vLLMOpenAIGeneratorConfig = cfg
    else:
        raise ValueError(f"Unknown GeneratorConfig.type: {cfg.type!r}")

    # LLaMA + Qwen both use the same runtime HF generator
    return HFCausalLMGenerator(
        model_name=sub.model_name,
        device=sub.device,
        temperature=sub.temperature,
        top_p=sub.top_p,
        top_k=sub.top_k,
        do_sample=sub.do_sample,
        max_new_tokens=sub.max_new_tokens,
        use_4bit=sub.use_4bit,
        hf_token=sub.hf_token,
    )
