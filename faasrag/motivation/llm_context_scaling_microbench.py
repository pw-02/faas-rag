"""
Microbenchmark: latency vs. input tokens for a HF causal LM.

What this script does:
- Loads your HFCausalLMGenerator
- Adds a generate_from_ids() method (so we can create exact-length token inputs)
- Sweeps over increasing prompt lengths (in tokens)
- Measures end-to-end generate() latency + completion tokens/sec
- (Optional) measures prefill-only forward latency too

Usage:
  python microbench_tokens.py

Notes:
- Make sure your model_name is correct. The one in your snippet looks wrong.
- Requires: transformers, torch, numpy
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from faasrag.core.generators import HFCausalLMGenerator
# -----------------------------
# Benchmark helpers
# -----------------------------
def _cuda_sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_input_ids_of_length(
    tokenizer: AutoTokenizer, target_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create an input_ids tensor of exactly target_len tokens.
    Uses a repeated filler string; tokenized once per call and then sliced.
    For tighter loops, you can precompute the filler IDs once and reuse.
    """
    filler = (" the" * 50000)  # enough tokens for long prompts
    ids = tokenizer(
        filler, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]

    if ids.numel() < target_len:
        raise ValueError(
            f"Filler produced only {ids.numel()} tokens; increase filler length."
        )

    input_ids = ids[:target_len].unsqueeze(0)  # [1, L]
    attention_mask = torch.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def percentile(xs: List[float], p: float) -> float:
    return float(np.percentile(np.array(xs), p))


@torch.no_grad()
def prefill_forward_time_s(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> float:
    """
    Prefill-only time: one forward pass over the prompt (no generation).
    """
    _cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    _cuda_sync_if_needed(device)
    return time.perf_counter() - t0


def bench(generator: HFCausalLMGenerator, 
          lengths: List[int], 
          reps: int = 10, 
          warmup: int = 1, 
          measure_prefill: bool = True,) -> List[Dict[str, Any]]:
    
    model_device = str(next(generator.model.parameters()).device)
    device_str = "cuda" if "cuda" in model_device else "cpu"

    # Warmup
    for _ in range(warmup):
        in_ids, am = make_input_ids_of_length(generator.tokenizer, lengths[0], model_device)
        _ = generator.generate_from_ids(in_ids, am)
        _cuda_sync_if_needed(device_str)

    results: List[Dict[str, Any]] = []

    for L in lengths:
        gen_times: List[float] = []
        prefill_times: List[float] = []
        comp_tokens_list: List[int] = []

        for _ in range(reps):
            in_ids, am = make_input_ids_of_length(generator.tokenizer, L, model_device)

            # Optional: prefill-only timing
            if measure_prefill:
                pt = prefill_forward_time_s(generator.model, in_ids, am, device_str)
                prefill_times.append(pt)

            # End-to-end generate timing
            _cuda_sync_if_needed(device_str)
            t0 = time.perf_counter()
            _text, prompt_tokens, completion_tokens, total_tokens = generator.generate_from_ids(in_ids, am)
            _cuda_sync_if_needed(device_str)
            t1 = time.perf_counter()

            dt = t1 - t0
            gen_times.append(dt)
            comp_tokens_list.append(completion_tokens)

        mean_gen = float(np.mean(gen_times))
        p50_gen = percentile(gen_times, 50)
        p95_gen = percentile(gen_times, 95)
        avg_comp = float(np.mean(comp_tokens_list))
        comp_tok_s = avg_comp / mean_gen if mean_gen > 0 else 0.0

        row: Dict[str, Any] = {
            "input_tokens": int(L),
            "mean_generate_s": mean_gen,
            "p50_generate_s": p50_gen,
            "p95_generate_s": p95_gen,
            "avg_completion_tokens": avg_comp,
            "completion_tok_per_s": comp_tok_s,
        }

        if measure_prefill and prefill_times:
            row["mean_prefill_s"] = float(np.mean(prefill_times))
            row["p50_prefill_s"] = percentile(prefill_times, 50)
            row["p95_prefill_s"] = percentile(prefill_times, 95)

            # -- estimate decode-only time and throughput ---
            decode_time_est = row["mean_generate_s"] - row["mean_prefill_s"]
            # guard against tiny/negative due to noise
            decode_time_est = max(decode_time_est, 1e-9)

            row["mean_decode_s_est"] = decode_time_est
            row["decode_tok_per_s_est"] = row["avg_completion_tokens"] / decode_time_est

        results.append(row)

        if measure_prefill and prefill_times:
            print(
                f"L={L:5d} | prefill={row['mean_prefill_s']:.3f}s "
                f"| gen={row['mean_generate_s']:.3f}s "
                f"| decode~={row['mean_decode_s_est']:.3f}s "
                f"| comp_tok/s(total)={row['completion_tok_per_s']:.1f} "
                f"| comp_tok/s(decode~)={row['decode_tok_per_s_est']:.1f}"
            )
        else:
            print(
                f"L={L:5d} | gen mean={mean_gen:.3f}s p95={p95_gen:.3f}s "
                f"| comp_tok/s={comp_tok_s:.1f}"
            )

    return results


# -----------------------------
# Config + main
# -----------------------------
@dataclass
class GeneratorConfig:
    model_name: str
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = False
    max_new_tokens: int = 128
    use_4bit: bool = False
    hf_token: Optional[str] = None


def main():
    # IMPORTANT: fix model_name. Your example looked malformed.
    # Use something you can actually load:
    # - "gpt2" (quick test)
    # - "meta-llama/Meta-Llama-3.1-8B-Instruct" (if you have access)
    config = GeneratorConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.0, #Only used if do_sample=True. For benchmarking, we usually want deterministic output, so temp=0 and do_sample=False.
        top_p=1.0, #Only used if sampling. Limits sampling to tokens covering top P probability mass.
        top_k=0, #Only used if sampling. Limits sampling to top K tokens by probability. 0 means no limit.
        do_sample=False, #False is greedy decoding (always pick max logit), which is faster and more stable for benchmarking.
        max_new_tokens=128, #Maximum number of tokens generated. Total compute = prompt tokens + max_new_tokens, so adjust accordingly for longer inputs.
        use_4bit=False,  #loads model in 4-bit quantized form (via bitsandbytes). Faster and less memory but can be less stable and slightly less performant
        hf_token=None, 
    )

    generator = HFCausalLMGenerator(
        model_name=config.model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        do_sample=config.do_sample,
        max_new_tokens=config.max_new_tokens,
        use_4bit=config.use_4bit,
        hf_token=config.hf_token,
    )

    # Sweep input lengths (tokens). Adjust upper bound to your GPU memory.
    lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]

    # Run benchmark
    results = bench(
        generator,
        lengths=lengths,
        reps=10,
        warmup=1,
        measure_prefill=True,
    )

    # ---- SAVE TO CSV ----
    df = pd.DataFrame(results)
    out_path = "llm_context_scaling_microbench.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved results to {out_path}\n")

if __name__ == "__main__":
    main()
