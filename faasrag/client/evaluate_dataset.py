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
    """
    Your dataset format:
      {"id": "...", "question": "...", "golden_answers": ["...", ...]}
    """
    q = (ex.get("question") or "").strip()
    golds = ex.get("golden_answers") or []
    if not isinstance(golds, list):
        golds = [str(golds)]
    golds = [str(a).strip() for a in golds if str(a).strip()]
    return q, golds


def evaluate(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    limit: Optional[int] = None,
    print_first_n: int = 10,
) -> Dict[str, float]:
    n = 0
    em_sum = 0.0
    f1_sum = 0.0

    subset = examples[:limit] if limit else examples

    for ex in subset:
        q, golds = get_question_and_answers(ex)
        if not q:
            continue

        out = pipeline.run(q)

        # IMPORTANT: use RAW answer for metrics (do not truncate).
        # Make sure your RagPipeline returns "raw_answer".
        pred = (out.get("raw_answer") or out.get("answer") or "").strip()

        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

        em_sum += float(em)
        f1_sum += float(f1)
        n += 1

        if n <= print_first_n:
            ex_id = ex.get("id", "")
            print("\n---")
            print(f"id: {ex_id}")
            print(f"Q: {q}")
            print(f"PRED: {pred}")
            print(f"GOLDS: {golds}")
            print(f"EM={em} F1={f1:.3f}")

        if n % 50 == 0:
            print(f"[{n}] running EM={em_sum/n:.3f} F1={f1_sum/n:.3f}")

    return {
        "n": n,
        "em": (em_sum / n) if n else 0.0,
        "f1": (f1_sum / n) if n else 0.0,
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset (jsonl)")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")
 

    # MODEL-ONLY baseline: set top_k=0 so retrieval is skipped.
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
    metrics = evaluate(pipeline, examples, limit=(args.limit if args.limit > 0 else None))
    print("\n==== FINAL ====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
