from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from faasrag.core.args import (GeneratorConfig, Llama3InstructGeneratorConfig, 
                               Qwen2_5InstructGeneratorConfig, SyntheticGeneratorConfig,
                                 vLLMOpenAIGeneratorConfig)

from dataclasses import dataclass
import time
import openai
import torch.nn.functional as F


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: Dict[str, Any]  # ttft_s, total_s, prefill_tps, decode_tps, finish_reason

# class SparseAddBiasProcessor(LogitsProcessor):
#     """
#     Adds alpha * bias[token_id] to logits during decoding.
#     bias is a sparse dict: token_id -> float.
#     """
#     def __init__(self, bias: dict[int, float], alpha: float, device: torch.device):
#         self.alpha = float(alpha)
#         if not bias:
#             self.token_ids = None
#             self.bias_vals = None
#             return

#         # store as tensors for fast scatter-add
#         self.token_ids = torch.tensor(list(bias.keys()), dtype=torch.long, device=device)
#         self.bias_vals = torch.tensor(list(bias.values()), dtype=torch.float32, device=device)

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         # scores: [batch, vocab]
#         if self.token_ids is None:
#             return scores
#         scores[:, self.token_ids] += self.alpha * self.bias_vals
#         return scores

class SparseAddBiasProcessor(LogitsProcessor):
    """
    Adds alpha * bias[token_id] to logits during decoding.

    Improvements:
      - max_steps: apply bias only for first N generated tokens
      - per_token_cap: clamp each bias value magnitude
      - ignore_eos: optionally avoid biasing EOS token
    """

    def __init__(
        self,
        bias: dict[int, float],
        alpha: float,
        device: torch.device,
        *,
        max_steps: int | None = None,
        per_token_cap: float | None = None,
        eos_token_id: int | None = None,
        ignore_eos: bool = True,
    ):
        self.alpha = float(alpha)
        self.max_steps = max_steps
        self.per_token_cap = per_token_cap
        self.eos_token_id = eos_token_id
        self.ignore_eos = ignore_eos

        self.step = 0  # track generation step

        if not bias:
            self.token_ids = None
            self.bias_vals = None
            return

        # Convert to tensors
        token_ids = list(bias.keys())
        bias_vals = list(bias.values())

        # Optionally remove EOS token from bias
        if ignore_eos and eos_token_id is not None:
            filtered = [
                (tid, val)
                for tid, val in zip(token_ids, bias_vals)
                if tid != eos_token_id
            ]
            if filtered:
                token_ids, bias_vals = zip(*filtered)
            else:
                token_ids, bias_vals = [], []

        if not token_ids:
            self.token_ids = None
            self.bias_vals = None
            return

        self.token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.bias_vals = torch.tensor(bias_vals, dtype=torch.float32, device=device)

        # Optional per-token cap
        if self.per_token_cap is not None:
            self.bias_vals = torch.clamp(
                self.bias_vals,
                min=-float(self.per_token_cap),
                max=float(self.per_token_cap),
            )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        scores: [batch, vocab]
        """
        if self.token_ids is None:
            return scores

        # If using max_steps and we've passed limit → stop biasing
        if self.max_steps is not None and self.step >= self.max_steps:
            return scores

        # Apply bias
        scores[:, self.token_ids] += self.alpha * self.bias_vals

        # Increment generation step
        self.step += 1

        return scores


# assume GenResult exists

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
            # common for Llama-family
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {}
        is_cuda = device.startswith("cuda")

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

        if is_cuda:
            kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, **kwargs
        )
        self.model.eval()

        # Cache special ids used for stopping (Llama 3.x often has <|eot_id|>)
        self.eos_id = self.tokenizer.eos_token_id
        self.eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if self.eot_id == self.tokenizer.unk_token_id:
            # some tokenizers may not know it; treat as missing
            self.eot_id = None

    def _move_inputs_to_model_device(self, inputs: dict) -> dict:
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    def _build_eos_token_ids(self) -> Union[int, List[int]]:
        # Prefer stopping at end-of-turn if available, otherwise eos
        if self.eot_id is not None and self.eot_id >= 0:
            # include both: sometimes eos appears before eot depending on tokenizer/model
            return [self.eot_id, self.eos_id]
        return self.eos_id
    
    
    def _chat_prompt_text(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


    @torch.no_grad()
    def generate(self, prompt: str) -> GenResult:
        """
        Backward-compatible: raw string prompt.
        NOTE: For Llama-3.1-Instruct, prefer generate_chat().
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = self._move_inputs_to_model_device(inputs)

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else 0,
            pad_token_id=self.eos_id,
            eos_token_id=self._build_eos_token_ids(),
        )

        prompt_tokens = inputs["input_ids"].shape[-1]
        total_tokens = out.shape[-1]
        completion_tokens = total_tokens - prompt_tokens
        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return GenResult(
            text=text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            metrics={},
        )

    @torch.no_grad()
    def generate_chat(self, messages: List[dict]) -> GenResult:
        """
        Preferred for instruction-tuned chat models (e.g., Meta-Llama-3.1-8B-Instruct).
        messages: [{"role":"system"|"user"|"assistant", "content": "..."}]
        """
        prompt = self._chat_prompt_text(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = self._move_inputs_to_model_device(inputs)
        
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else 0,
            pad_token_id=self.eos_id,
            eos_token_id=self._build_eos_token_ids(),
        )

        prompt_tokens = inputs["input_ids"].shape[-1]
        total_tokens = out.shape[-1]
        completion_tokens = total_tokens - prompt_tokens

        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Optional: clamp to first line to avoid “Answer: … user: …” leakage
        if "\n" in text:
            text = text.splitlines()[0].strip()

        return GenResult(
            text=text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            metrics={},
        )
    
    def generate_chat_with_logit_bias(
        self,
        messages: List[dict],
        bias: Dict[int, float],
        alpha: float = 2.0,
        clamp_first_line: bool = True,
        max_new_tokens: Optional[int] = None,
        *,
        prompt_max_length: Optional[int] = None,
        bias_steps: Optional[int] = None,   # <-- NEW: only bias first N generated tokens
    ) -> GenResult:
        """
        Stage-2 generation: normal chat generation, but with a sparse logit bias applied
        during decoding to nudge the model toward tokens mined from retrieved passages.

        messages:
        Chat messages [{"role": "...", "content": "..."}] formatted via chat template.
        bias:

        Dict[token_id -> bias_value]. bias_value is typically small (0..2). We multiply by alpha.

        alpha:
        Global strength knob. Effective bias is (alpha * bias_value) added to logits.

        prompt_max_length:
        Explicit prompt truncation length (strongly recommended to avoid silent truncation behavior).

        bias_steps:
        If set, apply logit bias only for the first N generated tokens.
        This often improves short-answer tasks by preventing bias from "dragging" later tokens.
        """
        # 1) Resolve generation length
        max_new = int(max_new_tokens) if max_new_tokens is not None else int(self.max_new_tokens)

        # 2) Chat template -> prompt string
        prompt = self._chat_prompt_text(messages)

        # 3) Tokenize prompt (explicit max_length avoids silent truncation surprises)
        tok_kwargs = dict(return_tensors="pt", truncation=True if prompt_max_length is not None else False)
        if prompt_max_length is not None:
            tok_kwargs["max_length"] = int(prompt_max_length)
        inputs = self.tokenizer(prompt, **tok_kwargs)
        inputs = self._move_inputs_to_model_device(inputs)
        device = inputs["input_ids"].device

        # 4) Build logits processor
        # If bias_steps is None: apply every step (current behavior).
        # If bias_steps is set: apply only early steps.
        if bias_steps is None:
            processors = LogitsProcessorList([SparseAddBiasProcessor(bias=bias, alpha=alpha, device=device)])
        else:
            processors = LogitsProcessorList([
                SparseAddBiasProcessor(
                    bias=bias,
                    alpha=alpha,
                    device=device,
                    max_steps=int(bias_steps),   # <-- you implement this in the processor
                )
            ])

        # 5) Generate
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else 0,
            pad_token_id=self.eos_id,
            eos_token_id=self._build_eos_token_ids(),
            logits_processor=processors,
        )

        # 6) Token accounting
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        total_tokens = int(out.shape[-1])
        completion_tokens = int(total_tokens - prompt_tokens)

        # 7) Decode completion
        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 8) Keep answer compact
        if clamp_first_line and "\n" in text:
            text = text.splitlines()[0].strip()

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metrics={
                "logit_bias_alpha": float(alpha),
                "logit_bias_tokens": int(len(bias) if bias else 0),
                "bias_steps": int(bias_steps) if bias_steps is not None else -1,
            },
        )

    @torch.no_grad()
    
    def score(
        self,
        prompt: str,
        completion: str,
        *,
        length_normalize: bool = False,) -> Tuple[float, int, int, int]:
        """
        Returns:
        (log P(completion | prompt), prompt_tokens, completion_tokens, total_tokens_scored)

        Note:
        - "completion_tokens" here means #tokens in the *completion string*.
        - total_tokens_scored = prompt_tokens + completion_tokens (for the full forward pass).
        """
        prompt = prompt or ""
        completion = completion or ""

        # Tokenize prompt alone
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        prompt_inputs = self._move_inputs_to_model_device(prompt_inputs)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        # Tokenize full (prompt + completion)
        full_text = prompt + completion
        full_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        full_inputs = self._move_inputs_to_model_device(full_inputs)
        input_ids = full_inputs["input_ids"]  # [1, T]
        attn = full_inputs.get("attention_mask", None)

        total_len = int(input_ids.shape[-1])
        completion_len = max(0, total_len - prompt_len)

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits  # [1, T, V]

        if prompt_len >= total_len:
            # nothing to score
            return 0.0, prompt_len, 0, prompt_len

        # Score only the completion portion
        target_ids = input_ids[:, prompt_len:]  # [1, L]
        pred_logits = logits[:, prompt_len - 1 : -1, :]  # [1, L, V]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L]
        seq_log_prob = float(token_log_probs.sum().item())

        if length_normalize:
            L = int(target_ids.shape[1])
            seq_log_prob = seq_log_prob / max(1, L)

        return seq_log_prob, prompt_len, completion_len, (prompt_len + completion_len)


    @torch.no_grad()
    def score_chat(
        self,
        messages: List[dict],
        completion: str,
        *,
        length_normalize: bool = False,
    ) -> Tuple[float, int, int, int]:
        """
        Same as score_chat(), but returns token counts too.
        """
        prompt = self._chat_prompt_text(messages)
        return self.score(prompt, completion, length_normalize=length_normalize)



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

    def generate(self, prompt: str) -> GenResult:
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

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metrics={},  # vLLM OpenAI-compatible endpoint doesn't currently return timing metrics in the response, so we leave this empty for now.
        )




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

    def generate(self, prompt: str) -> GenResult:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        text = f"{self.response_prefix} {prompt}"

        prompt_tokens = self._approx_tokens(prompt)
        completion_tokens = self._approx_tokens(text) - prompt_tokens
        completion_tokens = max(0, completion_tokens)
        total_tokens = prompt_tokens + completion_tokens

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metrics={},
        )

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
