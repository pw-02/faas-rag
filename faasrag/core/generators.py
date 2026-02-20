from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol

import openai
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor, LogitsProcessorList

from faasrag.core.args import (
    GeneratorConfig,
    Llama3InstructGeneratorConfig,
    Qwen2_5InstructGeneratorConfig,
    SyntheticGeneratorConfig,
    vLLMOpenAIGeneratorConfig,
)


# =========================
# Shared result type
# =========================
@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    metrics: Dict[str, Any]  # e.g. ttft_s, total_s, prefill_tps, decode_tps, finish_reason


# # =========================
# # Logits processor
# # =========================
# class SparseAddBiasProcessor(LogitsProcessor):
#     """
#     Adds alpha * bias[token_id] to logits during decoding.

#     Options:
#       - max_steps: apply only for first N generated tokens
#       - per_token_cap: clamp each bias value magnitude
#       - ignore_eos: optionally avoid biasing EOS token
#     """

#     def __init__(
#         self,
#         *,
#         bias: Dict[int, float],
#         logit_bias_strength: float, #Global strength multiplier.
#         device: torch.device,
#         max_steps: Optional[int] = None, #Only apply bias during first N generated tokens. Early tokens determine the direction of the answer.
#         per_token_cap: Optional[float] = None, #Clamp individual bias magnitudes.
#         eos_token_id: Optional[int] = None, #Avoid biasing EOS token.
#         ignore_eos: bool = True, #Avoid biasing EOS token.
#     ):
#         self.alpha = float(logit_bias_strength)
#         self.max_steps = max_steps
#         self._steps_seen = 0  # safe because we create a new processor per generate call

#         if not bias:
#             self.token_ids = None
#             self.bias_vals = None
#             return

#         items = list(bias.items())
#         if ignore_eos and eos_token_id is not None:
#             items = [(tid, val) for tid, val in items if tid != eos_token_id]

#         if not items:
#             self.token_ids = None
#             self.bias_vals = None
#             return

#         token_ids, bias_vals = zip(*items)
#         bias_vals_t = torch.tensor(bias_vals, dtype=torch.float32, device=device)

#         if per_token_cap is not None:
#             cap = float(per_token_cap)
#             bias_vals_t = torch.clamp(bias_vals_t, -cap, cap)

#         self.token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
#         self.bias_vals = bias_vals_t

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         if self.token_ids is None:
#             return scores

#         if self.max_steps is not None and self._steps_seen >= self.max_steps:
#             return scores

#         scores[:, self.token_ids] += self.alpha * self.bias_vals
#         self._steps_seen += 1
#         return scores


import torch
import torch.nn.functional as F
from transformers import LogitsProcessor
from typing import Dict, Optional




class GatedSparseAddBiasProcessor(LogitsProcessor):
    """
    Adds alpha * bias[token_id] to logits during decoding, but only when a confidence criterion is met.
    Good starting thresholds (very model/task dependent, but useful ballparks):

    | Gate Mode | What It Measures         | Mathematical Intuition | “Uncertain” When…            | Typical Threshold | Best Used For                  |
    | --------- | ------------------------ | ---------------------- | ---------------------------- | ----------------- | ------------------------------ |
    | `pmax`    | Peak probability         | max(softmax(logits))   | Top token probability is low | 0.25 – 0.45       | Simple global confidence       |
    | `margin`  | Decision gap             | logit₁ − logit₂        | Top two tokens are close     | 1.0 – 3.0         | Detecting ambiguity            |
    | `entropy` | Distribution flatness    | −Σ p log p             | Distribution is flat         | 2.0 – 3.5         | General uncertainty            |
    | `pbias`   | Alignment w/ bias tokens | Σ p(bias_tokens)       | Bias tokens have low mass    | 0.05 – 0.20       | RAG steering / token targeting |

    """
    def __init__(
        self,
        *,
        bias: Dict[int, float],
        logit_bias_strength: float,
        device: torch.device,
        max_steps: Optional[int] = None,
        per_token_cap: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        ignore_eos: bool = True,
        # gating config
        gate_mode: str = "pmax",   # "pmax" | "entropy" | "margin" | "pbias"
        gate_threshold: float = 10,  # meaning depends on mode
        gate_temperature: float = 1.0, # compute confidence at temp=1 for stability
        gate_topk: int = 50,          # approximate entropy on top-k for speed
        enable_gating: bool = True

    ):
        self.alpha = float(logit_bias_strength)
        self.max_steps = max_steps
        self._steps_seen = 0

        self.gate_mode = gate_mode
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.gate_topk = int(gate_topk)
        self.enable_gating = bool(enable_gating)

        items = list(bias.items())
        if ignore_eos and eos_token_id is not None:
            items = [(tid, val) for tid, val in items if tid != eos_token_id]

        if not items:
            self.token_ids = None
            self.bias_vals = None
            return

        token_ids, bias_vals = zip(*items)
        bias_vals_t = torch.tensor(bias_vals, dtype=torch.float32, device=device)
        if per_token_cap is not None:
            cap = float(per_token_cap)
            bias_vals_t = torch.clamp(bias_vals_t, -cap, cap)

        self.token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        self.bias_vals = bias_vals_t

    def _confidence(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores: [batch, vocab]
        s = scores / max(1e-6, self.gate_temperature)

        if self.gate_mode == "pmax":
            p = F.softmax(s, dim=-1)
            return p.max(dim=-1).values  # higher = more confident

        if self.gate_mode == "margin":
            top2 = torch.topk(s, k=2, dim=-1).values
            return top2[:, 0] - top2[:, 1]  # higher = more confident

        if self.gate_mode == "pbias":
            # probability mass on the biased tokens
            p = F.softmax(s, dim=-1)
            return p.index_select(dim=-1, index=self.token_ids).sum(dim=-1)  # higher = more aligned

        # default: entropy (approx on top-k for speed)
        k = min(self.gate_topk, s.size(-1))
        vals, idx = torch.topk(s, k=k, dim=-1)
        p = F.softmax(vals, dim=-1)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(dim=-1)  # higher = less confident
        return ent

    def __call__(self, input_ids, scores):
        if self.token_ids is None:
            return scores

        if self.max_steps is not None and self._steps_seen >= self.max_steps:
            return scores

        #If gating disabled → behave like original SparseAddBiasProcessor
        if not self.enable_gating:
            scores[:, self.token_ids] += self.alpha * self.bias_vals
            self._steps_seen += 1
            return scores
        
        # Otherwise: gated path
        conf = self._confidence(scores)
        if self.gate_mode == "entropy":
            apply = conf > self.gate_threshold
        else:
            apply = conf < self.gate_threshold

        if apply.any():
            scores[apply, self.token_ids] += self.alpha * self.bias_vals

        self._steps_seen += 1
        return scores




# =========================
# HF generator
# =========================
class HFCausalLMGenerator:
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

        is_cuda = device.startswith("cuda")
        kwargs: Dict[str, Any] = {}

        if self.use_4bit:
            if not is_cuda:
                raise ValueError("use_4bit=True is intended for CUDA devices only.")
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception as e:
                raise RuntimeError("use_4bit=True requires bitsandbytes (`pip install bitsandbytes`).") from e
        else:
            kwargs["torch_dtype"] = torch.float16 if is_cuda else torch.float32

        if is_cuda:
            kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, **kwargs).eval()

        # stopping ids
        self.eos_id = self.tokenizer.eos_token_id
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.eot_id = None if eot == self.tokenizer.unk_token_id else eot

    def _move_to_model_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    def _eos_token_ids(self) -> Union[int, List[int]]:
        return [self.eot_id, self.eos_id] if self.eot_id is not None else self.eos_id

    def _chat_prompt(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_from_prompt(
        self,
        *,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        prompt_max_length: Optional[int] = None,
        clamp_first_line: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> GenResult:
        tok_kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if prompt_max_length is not None:
            tok_kwargs.update({"truncation": True, "max_length": int(prompt_max_length)})
        else:
            tok_kwargs.update({"truncation": True})

        inputs = self._move_to_model_device(self.tokenizer(prompt, **tok_kwargs))

        out = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens) if max_new_tokens is not None else self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else 0.0,
            top_p=self.top_p if self.do_sample else 1.0,
            top_k=self.top_k if self.do_sample else 0,
            pad_token_id=self.eos_id,
            eos_token_id=self._eos_token_ids(),
            logits_processor=logits_processor,
        )

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        total_tokens = int(out.shape[-1])
        completion_tokens = total_tokens - prompt_tokens

        gen_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if clamp_first_line and "\n" in text:
            text = text.splitlines()[0].strip()

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metrics=metrics or {},
        )

    @torch.no_grad()
    def generate(self, prompt: str) -> GenResult:
        return self._generate_from_prompt(prompt=prompt)

    @torch.no_grad()
    def generate_chat(self, messages: List[dict]) -> GenResult:
        return self._generate_from_prompt(prompt=self._chat_prompt(messages), clamp_first_line=True)

    @torch.no_grad()
    def generate_chat_with_logit_bias(
        self,
        messages: List[dict],
        bias: Dict[int, float],
        *,
        logit_bias_strength: float = 2.0,
        max_bias_steps: Optional[int] = None,
        clamp_first_line: bool = True,
        max_new_tokens: Optional[int] = None,
        prompt_max_length: Optional[int] = None,
        per_token_cap: Optional[float] = None,
        ignore_eos: bool = True,
    ) -> GenResult:
        prompt = self._chat_prompt(messages)
        device = next(self.model.parameters()).device

        processors: Optional[LogitsProcessorList]
        if bias:
            proc = GatedSparseAddBiasProcessor(
                bias=bias,
                logit_bias_strength=logit_bias_strength,
                device=device,
                max_steps=max_bias_steps,
                per_token_cap=per_token_cap,
                eos_token_id=self.eos_id,
                ignore_eos=ignore_eos,
            )
            processors = LogitsProcessorList([proc])
        else:
            processors = None

        return self._generate_from_prompt(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            prompt_max_length=prompt_max_length,
            logits_processor=processors,
            clamp_first_line=clamp_first_line,
            metrics={
                "logit_bias_strength": float(logit_bias_strength),
                "logit_bias_tokens": int(len(bias) if bias else 0),
                "max_bias_steps": int(max_bias_steps) if max_bias_steps is not None else -1,
            },
        )

    @torch.no_grad()
    def score(
        self,
        prompt: str,
        completion: str,
        *,
        length_normalize: bool = False,
    ) -> Tuple[float, int, int, int]:
        prompt = prompt or ""
        completion = completion or ""

        prompt_inputs = self._move_to_model_device(self.tokenizer(prompt, return_tensors="pt", truncation=True))
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        full_inputs = self._move_to_model_device(
            self.tokenizer(prompt + completion, return_tensors="pt", truncation=True)
        )
        input_ids = full_inputs["input_ids"]
        attn = full_inputs.get("attention_mask")

        total_len = int(input_ids.shape[-1])
        completion_len = max(0, total_len - prompt_len)
        if prompt_len >= total_len:
            return 0.0, prompt_len, 0, prompt_len

        logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        target_ids = input_ids[:, prompt_len:]
        pred_logits = logits[:, prompt_len - 1 : -1, :]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_prob = float(token_log_probs.sum().item())

        if length_normalize:
            L = int(target_ids.shape[1])
            seq_log_prob /= max(1, L)

        return seq_log_prob, prompt_len, completion_len, (prompt_len + completion_len)

    @torch.no_grad()
    def score_chat(
        self,
        messages: List[dict],
        completion: str,
        *,
        length_normalize: bool = False,
    ) -> Tuple[float, int, int, int]:
        return self.score(self._chat_prompt(messages), completion, length_normalize=length_normalize)
    

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




# =========================
# vLLM OpenAI-compatible generator (streaming + non-streaming)
# =========================
class VLLMOpenAIGenerator:
    """
    Calls a vLLM OpenAI-compatible server (chat.completions).
    Supports streaming (for TTFT / throughput metrics) and non-streaming.
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
        stream: bool = False,
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.stream = bool(stream)

    def generate(self, prompt: str) -> GenResult:
        if not self.stream:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()

            usage = getattr(resp, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

            return GenResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                metrics={},
            )

        # streaming path
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

                delta = getattr(choice.delta, "content", None)
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_text += delta

            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

        end_time = time.time()

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )

        ttft_s = (first_token_time - start_time) if first_token_time else None
        total_s = end_time - start_time

        prefill_tps = (prompt_tokens / ttft_s) if (ttft_s and ttft_s > 0 and prompt_tokens > 0) else None
        decode_tps = None
        if ttft_s is not None:
            decode_time_s = max(total_s - ttft_s, 1e-9)
            decode_tps = (completion_tokens / decode_time_s) if completion_tokens > 0 else None

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


# =========================
# Synthetic generator
# =========================
class SyntheticGenerator:
    def __init__(self, *, sleep_seconds: float, response_prefix: str):
        self.sleep_seconds = float(sleep_seconds)
        self.response_prefix = response_prefix

    @staticmethod
    def _approx_tokens(s: str) -> int:
        s = s or ""
        return max(1, len(s) // 4) if s else 0

    def generate(self, prompt: str) -> GenResult:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        text = f"{self.response_prefix} {prompt}".strip()

        prompt_tokens = self._approx_tokens(prompt)
        total_tokens = self._approx_tokens(text)
        completion_tokens = max(0, total_tokens - prompt_tokens)

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(prompt_tokens + completion_tokens),
            metrics={},
        )


# =========================
# Generator protocol + union
# =========================
class GeneratorLike(Protocol):
    def generate(self, prompt: str) -> GenResult: ...


Generator = Union[HFCausalLMGenerator, VLLMOpenAIGenerator, SyntheticGenerator]


# =========================
# Builder
# =========================
def build_generator(cfg: GeneratorConfig) -> Generator:
    """
    Build a generator from the tagged-union GeneratorConfig wrapper.

    Supported types:
      - "synthetic"
      - "llama3_instruct"
      - "qwen2_5_instruct"
      - "vllm"
    """
    if cfg.type == "synthetic":
        sub: SyntheticGeneratorConfig = cfg
        return SyntheticGenerator(
            sleep_seconds=sub.sleep_time,
            response_prefix=sub.response_prefix,
        )

    if cfg.type in {"llama3_instruct", "qwen2_5_instruct"}:
        # both use HF generator runtime
        sub: Union[Llama3InstructGeneratorConfig, Qwen2_5InstructGeneratorConfig] = cfg
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

    if cfg.type == "vllm":
        sub: vLLMOpenAIGeneratorConfig = cfg
        # If your config includes a "stream" bool, it will be used; otherwise default False.
        stream = bool(getattr(sub, "stream", False))
        return VLLMOpenAIGenerator(
            base_url=sub.base_url,
            api_key=sub.api_key,
            model=sub.model,
            temperature=sub.temperature,
            max_tokens=sub.max_tokens,
            timeout_s=sub.timeout_s,
            stream=stream,
        )

    raise ValueError(f"Unknown GeneratorConfig.type: {cfg.type!r}")
