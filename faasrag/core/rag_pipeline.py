from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import hydra

from faasrag.core.args import RagServiceConfig
from faasrag.core.utils import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    append_csv_row,
)

from faasrag.core.rag_pipeline import RagPipeline


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def get_question_and_answers(ex: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = (ex.get("question") or "").strip()
    golds = ex.get("golden_answers") or []
    if not isinstance(golds, list):
        golds = [str(golds)]
    golds = [str(a).strip() for a in golds if str(a).strip()]
    return q, golds


def safe_get(d: Dict[str, Any], path: List[str], default=0.0):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def evaluate(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    limit: Optional[int] = None,
    print_first_n: int = 10,
    out_csv: Optional[str] = None,
) -> Dict[str, float]:
    n = 0
    em_sum = 0.0
    f1_sum = 0.0

    # token totals
    prompt_tok_sum = 0
    completion_tok_sum = 0
    total_tok_sum = 0

    # timing totals (seconds)
    embed_sum = 0.0
    ann_sum = 0.0
    docstore_sum = 0.0
    prompt_build_sum = 0.0
    decode_sum = 0.0
    ttft_sum = 0.0
    total_s_sum = 0.0

    subset = examples[:limit] if limit else examples

    for ex in subset:
        ex_id = ex.get("id", "")
        q, golds = get_question_and_answers(ex)
        if not q:
            continue

        out = pipeline.run(q)

        pred = (out.get("raw_answer") or out.get("answer") or "").strip()

        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

        # tokens
        prompt_tokens = int(out.get("prompt_tokens", 0) or 0)
        completion_tokens = int(out.get("completion_tokens", 0) or 0)
        total_tokens = int(out.get("total_tokens", 0) or 0)

        # timings
        timings = out.get("timings_s") or {}
        embed_s = float(timings.get("embed_s", 0.0) or 0.0)
        ann_s = float(timings.get("ann_s", 0.0) or 0.0)
        docstore_s = float(timings.get("docstore_s", 0.0) or 0.0)
        prompt_s = float(timings.get("prompt_s", 0.0) or 0.0)
        decode_s = float(timings.get("decode_s", 0.0) or 0.0)

        ttft_s = float(timings.get("ttft_s", 0.0) or 0.0)
        total_s = float(timings.get("total_s", 0.0) or 0.0)

        retrieved_doc_ids = out.get("retrieved_doc_ids") or []
        finish_reason = out.get("finish_reason", "")

        cache_used = int(bool(out.get("cache_used", False)))
        cache_hits = int(out.get("cache_hits", 0) or 0)
        cache_misses = int(out.get("cache_misses", 0) or 0)

        # accumulate
        em_sum += float(em)
        f1_sum += float(f1)
        n += 1

        prompt_tok_sum += prompt_tokens
        completion_tok_sum += completion_tokens
        total_tok_sum += total_tokens

        embed_sum += embed_s
        ann_sum += ann_s
        docstore_sum += docstore_s
        prompt_build_sum += prompt_s
        decode_sum += decode_s
        ttft_sum += ttft_s
        total_s_sum += total_s

        if n <= print_first_n:
            print("\n---")
            print(f"id: {ex_id}")
            print(f"Q: {q}")
            print(f"PRED: {pred}")
            print(f"GOLDS: {golds}")
            print(f"EM={em} F1={f1:.3f}")
            print(f"tokens: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
            print(f"timings_s: total={total_s:.3f} ttft={ttft_s:.3f} decode={decode_s:.3f}")

        if n % 50 == 0:
            print(f"[{n}] running EM={em_sum/n:.3f} F1={f1_sum/n:.3f} total_s={total_s_sum/n:.3f}")

        # write per-example CSV row
        append_csv_row(out_csv, {
            "id": ex_id,
            "question": q,
            "prediction": pred,
            "golden_answers": json.dumps(golds, ensure_ascii=False),

            "em": float(em),
            "f1": float(f1),

            "top_k": int(getattr(pipeline, "top_k", 0)),
            "retrieved_count": len(retrieved_doc_ids),
            "retrieved_doc_ids": json.dumps(retrieved_doc_ids),

            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,

            "embed_s": embed_s,
            "ann_s": ann_s,
            "docstore_s": docstore_s,
            "prompt_build_s": prompt_s,
            "decode_s": decode_s,
            "ttft_s": ttft_s,
            "total_s": total_s,

            "cache_used": cache_used,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        })

    # final means
    def mean(x: float) -> float:
        return x / n if n else 0.0

    return {
        "n": n,
        "em": mean(em_sum),
        "f1": mean(f1_sum),

        "mean_prompt_tokens": mean(prompt_tok_sum),
        "mean_completion_tokens": mean(completion_tok_sum),
        "mean_total_tokens": mean(total_tok_sum),

        "mean_embed_s": mean(embed_sum),
        "mean_ann_s": mean(ann_sum),
        "mean_docstore_s": mean(docstore_sum),
        "mean_prompt_build_s": mean(prompt_build_sum),
        "mean_decode_s": mean(decode_sum),
        "mean_ttft_s": mean(ttft_sum),
        "mean_total_s": mean(total_s_sum),
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset (jsonl)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick runs")
    parser.add_argument("--out_csv", type=str, default="", help="Optional CSV path for per-example results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")

    pipeline = RagPipeline(
        generator_cfg=cfg.generator,
        embedder_cfg=cfg.embedder,
        index_cfg=cfg.index,
        docstore_cfg=cfg.docstore,
        docstore_backend=cfg.docstore_backend,
        artifact_dir=cfg.artifact_dir,
        prompt_build_method=cfg.prompt_build_method,
        max_ctx_chars=cfg.max_ctx_chars,
        cache_cfg=None,
        top_k=0,  # model-only
        logger=logger,
        retrieve_only=False,
        always_log_results=False,
    )

    examples = load_jsonl(args.data)
    metrics = evaluate(
        pipeline,
        examples,
        limit=(args.limit if args.limit > 0 else None),
        out_csv=(args.out_csv.strip() or None),
    )

    print("\n==== FINAL ====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
