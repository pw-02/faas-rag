"""
concurrency_bench_vllm.py

Quick concurrency benchmark for a vLLM OpenAI-compatible server.
- Sends N requests concurrently (async)
- Measures TTFT, total latency, tokens/sec, success rate
- Optional fixed-QPS mode (approx) via semaphore + pacing

Usage:
  pip install openai

  python concurrency_bench_vllm.py \
    --base-url http://134.197.95.82:8000/v1 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --concurrency 16 \
    --requests 200 \
    --max-tokens 64 \
    --prompt "Answer with one word: hello"

Optional:
  --qps 10   # pace launches to ~10 requests/sec (still concurrent)
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import openai


@dataclass
class Result:
    ok: bool
    status: str
    ttft_s: Optional[float]
    total_s: Optional[float]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def p50(values: List[float]) -> float:
    return statistics.median(values)


def p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(round(0.95 * (len(s) - 1)))
    return s[idx]


async def one_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Result:
    start = time.perf_counter()
    first_token_t: Optional[float] = None
    full_text = ""
    usage = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    full_text += delta

            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage

        end = time.perf_counter()
        ttft_s = (first_token_t - start) if first_token_t else None
        total_s = end - start

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )

        return Result(
            ok=True,
            status="ok",
            ttft_s=ttft_s,
            total_s=total_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    except Exception as e:
        return Result(
            ok=False,
            status=f"{type(e).__name__}: {e}",
            ttft_s=None,
            total_s=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )


async def run_bench(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    num_requests: int,
    qps: Optional[float],
    timeout_s: float,
) -> List[Result]:
    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
    )

    sem = asyncio.Semaphore(concurrency)

    async def worker(i: int) -> Result:
        async with sem:
            return await one_request(client, model, prompt, max_tokens, temperature)

    tasks = []
    t0 = time.perf_counter()

    # launch with optional pacing
    for i in range(num_requests):
        tasks.append(asyncio.create_task(worker(i)))
        if qps and qps > 0:
            await asyncio.sleep(1.0 / qps)

    results = await asyncio.gather(*tasks)
    t1 = time.perf_counter()
    wall = t1 - t0
    ok = sum(1 for r in results if r.ok)
    print(f"\nFinished: ok={ok}/{num_requests}  wall_time={wall:.3f}s  achieved_rps={num_requests / wall:.2f}")

    return results


def summarize(results: List[Result]) -> None:
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        print("No successful requests.")
        print("Example error:", results[0].status if results else "n/a")
        return

    ttfts = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    totals = [r.total_s for r in ok_results if r.total_s is not None]
    prompt_toks = [r.prompt_tokens for r in ok_results]
    comp_toks = [r.completion_tokens for r in ok_results]

    # Aggregate tokens/sec (decode) using sum(tokens)/sum(decode_time)
    decode_times = []
    for r in ok_results:
        if r.ttft_s is None or r.total_s is None:
            continue
        decode_times.append(max(r.total_s - r.ttft_s, 1e-9))

    total_comp = sum(comp_toks)
    total_decode_time = sum(decode_times) if decode_times else float("nan")
    agg_decode_tps = (total_comp / total_decode_time) if total_decode_time and total_decode_time > 0 else float("nan")

    print("\n=== Summary (successful requests) ===")
    print(f"count: {len(ok_results)} / {len(results)}")
    if ttfts:
        print(f"TTFT  p50={p50(ttfts):.3f}s  p95={p95(ttfts):.3f}s  avg={statistics.mean(ttfts):.3f}s")
    else:
        print("TTFT  n/a (no TTFT data)")
    print(f"Total p50={p50(totals):.3f}s  p95={p95(totals):.3f}s  avg={statistics.mean(totals):.3f}s")
    print(f"Prompt tokens   avg={statistics.mean(prompt_toks):.1f}")
    print(f"Completion toks avg={statistics.mean(comp_toks):.1f}")
    print(f"Aggregate decode tokens/sec ≈ {agg_decode_tps:.2f}")

    # show a few errors if any
    errors = [r.status for r in results if not r.ok]
    if errors:
        print("\nErrors (sample up to 5):")
        for e in errors[:5]:
            print(" -", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default='http://134.197.95.82:8000/v1', help="e.g. http://134.197.95.82:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--prompt", default="Answer with one word: hello")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--qps", type=float, default=None, help="Optional launch pacing, e.g. 10")
    ap.add_argument("--timeout-s", type=float, default=120.0)
    args = ap.parse_args()

    results = asyncio.run(
        run_bench(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            num_requests=args.requests,
            qps=args.qps,
            timeout_s=args.timeout_s,
        )
    )
    summarize(results)


if __name__ == "__main__":
    main()
