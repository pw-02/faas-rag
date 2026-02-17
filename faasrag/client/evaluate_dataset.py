# =========================
# FILE 2: evaluate_dataset.py
# Cleaned evaluation with:
# - correct mode validation
# - clean dispatch
# - stage2 CLI knobs
# - consistent logging/CSV schema
# =========================
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
from tqdm.auto import tqdm

from faasrag.core.args import RagServiceConfig
from faasrag.core.utils import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    append_csv_row,
    normalize_answer,
    parse_float_list
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


def gold_in_candidates(candidates: List[Dict[str, Any]], golds: List[str]) -> bool:
    if not candidates or not golds:
        return False
    cand_norm = {normalize_answer(str(c.get("candidate", ""))) for c in candidates}
    for g in golds:
        if normalize_answer(g) in cand_norm:
            return True
    return False


def model_selected_gold(best_candidate: str, golds: List[str]) -> bool:
    if not best_candidate or not golds:
        return False
    b = normalize_answer(best_candidate)
    return any(b == normalize_answer(g) for g in golds)






def evaluate(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    *,
    mode: str,
    limit: Optional[int] = None,
    print_first_n: int = 10,
    out_csv: Optional[str] = None,
    tqdm_update_every: int = 10,
    # stage-1 knobs
    stage1_max_candidates: int = 40,
    stage1_score_top_n: int = 20,
    stage1_length_normalize: bool = True,
    stage1_alpha_prior: float = 0.0,
    # stage-2 knobs
    stage2_alpha: float = 0.8,
    stage2_hybrid_prompt: bool = False,
    stage2_max_phrases: int = 30,
    save_to_file = False,
) -> Dict[str, float]:
    mode = (mode or "").strip().lower()
    valid = {"llm", "prompt_rag", "logit_rag_stage1", "logit_rag_stage2"}
    if mode not in valid:
        raise ValueError(f"Invalid --mode {mode}. Choose: {', '.join(sorted(valid))}")

    # core metrics
    n = 0
    em_sum = 0.0
    f1_sum = 0.0

    # token totals
    prompt_tok_sum = 0
    completion_tok_sum = 0
    total_tok_sum = 0

    # timing totals
    total_s_sum = 0.0
    decode_sum = 0.0
    embed_sum = 0.0
    ann_sum = 0.0
    docstore_sum = 0.0
    prompt_build_sum = 0.0

    # stage1 timing totals
    stage1_mine_sum = 0.0
    stage1_score_sum = 0.0

    # stage1 extra reporting
    stage1_num_scored_sum = 0
    stage1_tok_per_cand_sum = 0.0
    stage1_tok_per_cand_n = 0

    # stage1 diagnostics
    cand_recall_hits = 0
    cand_recall_total = 0
    select_hits = 0
    select_total = 0

    # stage2 diagnostics
    stage2_n = 0
    stage2_mined_hit_sum = 0
    stage2_gold_in_mined_sum = 0

    # NEW: stage2 oracle + selection-given-oracle
    stage2_oracle_hits = 0
    stage2_oracle_total = 0
    stage2_select_hits_given_oracle = 0
    stage2_select_total_given_oracle = 0

    subset = examples[:limit] if limit else examples
    pbar = tqdm(subset, total=len(subset), desc=f"Evaluating ({mode})", dynamic_ncols=True)

    if out_csv and save_to_file:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("")

    def mean(x: float) -> float:
        return x / n if n else 0.0

    for ex in pbar:
        ex_id = ex.get("id", "")
        q, golds = get_question_and_answers(ex)
        if not q:
            continue

        # Dispatch
        if mode == "logit_rag_stage1":
            out = pipeline.run_logit_rag_stage1(
                q,
                max_candidates=stage1_max_candidates,
                score_top_n=stage1_score_top_n,
                length_normalize=stage1_length_normalize,
                alpha_prior=stage1_alpha_prior,
            )
        elif mode == "logit_rag_stage2":
            out = pipeline.run_logit_rag_stage2(
                q,
                alpha=stage2_alpha,
                hybrid_prompt=stage2_hybrid_prompt,
                max_phrases_for_bias=stage2_max_phrases,
            )
        elif mode == "prompt_rag":
            out = pipeline.run_prompt_rag(q)
        elif mode == "llm":
            out = pipeline.run_prompt_rag(q)
        else:
            raise RuntimeError("unreachable")

        # Prediction + metrics
        pred = (out.get("raw_answer") or out.get("answer") or "").strip()
        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

        messages = out.get("messages", "")
        timings = out.get("timings_s") or {}
        embed_s = float(timings.get("embed_s", 0.0) or 0.0)
        ann_s = float(timings.get("ann_s", 0.0) or 0.0)
        docstore_s = float(timings.get("docstore_s", 0.0) or 0.0)
        prompt_s = float(timings.get("prompt_s", 0.0) or 0.0)
        decode_s = float(timings.get("decode_s", 0.0) or 0.0)
        total_s = float(timings.get("total_s", 0.0) or 0.0)

        mine_s = float(timings.get("candidate_mine_s", 0.0) or 0.0)
        cand_score_s = float(timings.get("candidate_score_s", 0.0) or 0.0)

        prompt_tokens = int(out.get("prompt_tokens", 0) or 0)
        completion_tokens = int(out.get("completion_tokens", 0) or 0)
        total_tokens = int(out.get("total_tokens", 0) or 0)

        retrieved_doc_ids = out.get("retrieved_doc_ids") or []

        # extra payload (stage1 stores candidates/best here)
        extra = out.get("extra") or {}

        # cache fields may be top-level (stage2) or inside extra (stage1)
        cache_used = out.get("cache_used", None)
        cache_hits = out.get("cache_hits", None)
        cache_misses = out.get("cache_misses", None)
        if cache_used is None:
            cache_used = extra.get("cache_used", False)
        if cache_hits is None:
            cache_hits = extra.get("cache_hits", 0)
        if cache_misses is None:
            cache_misses = extra.get("cache_misses", 0)

        # Stage1 candidates/best (robust)
        candidates = out.get("candidates") or extra.get("candidates") or []
        best = out.get("best_candidate") or extra.get("best_candidate") or ""

        num_candidates_scored = len(candidates) if isinstance(candidates, list) else 0

        mean_tokens_per_candidate = 0.0
        if mode == "logit_rag_stage1" and num_candidates_scored > 0:
            mean_tokens_per_candidate = float(total_tokens) / float(num_candidates_scored)
            stage1_num_scored_sum += int(num_candidates_scored)
            stage1_tok_per_cand_sum += float(mean_tokens_per_candidate)
            stage1_tok_per_cand_n += 1

        # Stage1 diagnostic flags
        has_gold = False
        selected_gold = False
        if mode == "logit_rag_stage1":
            cand_recall_total += 1
            has_gold = gold_in_candidates(candidates, golds)
            cand_recall_hits += int(has_gold)
            if has_gold:
                select_total += 1
                selected_gold = model_selected_gold(best, golds)
                select_hits += int(selected_gold)

        # Stage2 diagnostic flags
        mined_hit = False
        gold_in_mined = False

        # NEW: stage2 oracle + selection-given-oracle
        oracle_em_from_mined = False
        selected_gold_given_oracle = False

        if mode == "logit_rag_stage2":
            stage2_n += 1

            mined = out.get("mined_candidates") or []
            mined_norm = [normalize_answer(str(x)) for x in mined]
            mined_norm = [x for x in mined_norm if x]

            pred_norm = normalize_answer(pred)
            mined_hit = any(m and (m in pred_norm) for m in mined_norm[:30])

            gold_norms = [normalize_answer(str(g)) for g in golds]
            gold_norms = [g for g in gold_norms if g]

            mined_set = set(mined_norm)
            gold_in_mined = any(g in mined_set for g in gold_norms)

            stage2_mined_hit_sum += int(mined_hit)
            stage2_gold_in_mined_sum += int(gold_in_mined)

            # -------- NEW METRICS --------
            stage2_oracle_total += 1
            oracle_em_from_mined = gold_in_mined  # exact match oracle, normalized
            stage2_oracle_hits += int(oracle_em_from_mined)

            if oracle_em_from_mined:
                stage2_select_total_given_oracle += 1
                selected_gold_given_oracle = any(pred_norm == g for g in gold_norms)
                stage2_select_hits_given_oracle += int(selected_gold_given_oracle)

        # Accumulate totals
        n += 1
        em_sum += float(em)
        f1_sum += float(f1)

        prompt_tok_sum += prompt_tokens
        completion_tok_sum += completion_tokens
        total_tok_sum += total_tokens

        embed_sum += embed_s
        ann_sum += ann_s
        docstore_sum += docstore_s
        prompt_build_sum += prompt_s
        decode_sum += decode_s
        total_s_sum += total_s

        stage1_mine_sum += mine_s
        stage1_score_sum += cand_score_s

        # Print debug
        if n <= print_first_n:
            tqdm.write("\n---")
            tqdm.write(f"mode: {mode}  id: {ex_id}")
            tqdm.write(f"Q: {q}")
            tqdm.write(f"PRED: {pred}")
            tqdm.write(f"GOLDS: {golds}")
            tqdm.write(f"EM={em} F1={f1:.3f}")
            tqdm.write(f"tokens: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
            tqdm.write(f"timings_s: total={total_s:.3f} decode={decode_s:.3f}")

            if mode == "logit_rag_stage1":
                tqdm.write(f"num_candidates_scored: {num_candidates_scored}")
                tqdm.write(f"mean_tokens_per_candidate: {mean_tokens_per_candidate:.2f}")
                tqdm.write(f"gold_in_candidates: {has_gold} selected_gold: {selected_gold}")
                tqdm.write(f"best_candidate: {best}")
                if candidates:
                    tqdm.write("top5_candidates: " + json.dumps(candidates[:5], ensure_ascii=False)[:600])

            if mode == "logit_rag_stage2":
                tqdm.write(
                    f"bias_tokens: {out.get('bias_tokens', 0)} alpha: {out.get('alpha', 0.0)} hybrid_prompt: {stage2_hybrid_prompt}"
                )
                tqdm.write(
                    f"mined_hit: {mined_hit} gold_in_mined: {gold_in_mined} "
                    f"oracle_em_from_mined: {oracle_em_from_mined} selected_gold_given_oracle: {selected_gold_given_oracle}"
                )
                mined = out.get("mined_candidates") or []
                if mined:
                    tqdm.write("mined[:10]: " + json.dumps(mined[:10], ensure_ascii=False)[:400])

        # Progress postfix
        if tqdm_update_every > 0 and n % tqdm_update_every == 0:
            postfix = {
                "EM": f"{em_sum/n:.3f}",
                "F1": f"{f1_sum/n:.3f}",
                "tok": f"{(total_tok_sum/n):.0f}",
                "tot_s": f"{(total_s_sum/n):.2f}",
            }
            if mode == "logit_rag_stage1":
                cand_rec = (cand_recall_hits / cand_recall_total) if cand_recall_total else 0.0
                sel_acc = (select_hits / select_total) if select_total else 0.0
                postfix["cand@"] = f"{cand_rec:.2f}"
                postfix["sel@"] = f"{sel_acc:.2f}"
                if stage1_tok_per_cand_n > 0:
                    postfix["tok/cand"] = f"{(stage1_tok_per_cand_sum / stage1_tok_per_cand_n):.1f}"
            if mode == "logit_rag_stage2" and stage2_n > 0:
                postfix["mined_hit"] = f"{(stage2_mined_hit_sum/stage2_n):.2f}"
                postfix["gold_in_m"] = f"{(stage2_gold_in_mined_sum/stage2_n):.2f}"
                # NEW postfix bits
                postfix["oracle"] = f"{(stage2_oracle_hits/stage2_oracle_total):.2f}" if stage2_oracle_total else "0.00"
                postfix["sel|oracle"] = (
                    f"{(stage2_select_hits_given_oracle/stage2_select_total_given_oracle):.2f}"
                    if stage2_select_total_given_oracle else "0.00"
                )
            pbar.set_postfix(postfix)

        # CSV row
        row = {
            "mode": mode,
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

            "embed_s": embed_s,
            "ann_s": ann_s,
            "docstore_s": docstore_s,
            "prompt_build_s": prompt_s,
            "decode_s": decode_s,
            "total_s": total_s,

            "candidate_mine_s": mine_s,
            "candidate_score_s": cand_score_s,

            "cache_used": int(bool(cache_used)),
            "cache_hits": int(cache_hits or 0),
            "cache_misses": int(cache_misses or 0),
        }

        if mode == "logit_rag_stage1":
            row["gold_in_candidates"] = int(has_gold)
            row["selected_gold_given_present"] = int(selected_gold)
            row["best_candidate"] = best
            row["num_candidates_scored"] = int(num_candidates_scored)
            row["mean_tokens_per_candidate"] = float(mean_tokens_per_candidate)

        if mode == "logit_rag_stage2":
            row["bias_tokens"] = int(out.get("bias_tokens", 0) or 0)
            row["alpha"] = float(out.get("alpha", 0.0) or 0.0)
            row["hybrid_prompt"] = int(bool(stage2_hybrid_prompt))
            row["mined_hit"] = int(mined_hit)
            row["gold_in_mined"] = int(gold_in_mined)
            # NEW per-row fields
            row["oracle_em_from_mined"] = int(oracle_em_from_mined)
            row["selected_gold_given_gold_in_mined"] = int(selected_gold_given_oracle)
        if save_to_file:
            append_csv_row(out_csv, row)

    summary: Dict[str, float] = {
        "mode": mode,
        "n": float(n),
        "em": mean(em_sum),
        "f1": mean(f1_sum),

        "mean_total_tokens": mean(total_tok_sum),

        "mean_embed_s": mean(embed_sum),
        "mean_ann_s": mean(ann_sum),
        "mean_docstore_s": mean(docstore_sum),
        "mean_prompt_build_s": mean(prompt_build_sum),
        "mean_decode_s": mean(decode_sum),
        "mean_total_s": mean(total_s_sum),

        "mean_candidate_mine_s": mean(stage1_mine_sum),
        "mean_candidate_score_s": mean(stage1_score_sum),
    }

    if mode == "logit_rag_stage1":
        summary["candidate_recall"] = (cand_recall_hits / cand_recall_total) if cand_recall_total else 0.0
        summary["selection_accuracy_given_present"] = (select_hits / select_total) if select_total else 0.0
        summary["mean_num_candidates_scored"] = (stage1_num_scored_sum / n) if n else 0.0
        summary["mean_tokens_per_candidate"] = (
            (stage1_tok_per_cand_sum / stage1_tok_per_cand_n) if stage1_tok_per_cand_n else 0.0
        )

    if mode == "logit_rag_stage2":
        summary["mined_hit_rate"] = (stage2_mined_hit_sum / stage2_n) if stage2_n else 0.0
        summary["gold_in_mined_rate"] = (stage2_gold_in_mined_sum / stage2_n) if stage2_n else 0.0

        # NEW: oracle + selection given oracle
        summary["oracle_em_from_mined"] = (stage2_oracle_hits / stage2_oracle_total) if stage2_oracle_total else 0.0
        summary["selection_accuracy_given_gold_in_mined"] = (
            (stage2_select_hits_given_oracle / stage2_select_total_given_oracle)
            if stage2_select_total_given_oracle else 0.0
        )
        summary["selection_total_where_gold_in_mined"] = float(stage2_select_total_given_oracle)

        summary["alpha"] = float(stage2_alpha)
        summary["max_phrases_for_bias"] = int(stage2_max_phrases)
        summary["hybrid_prompt"] = int(bool(stage2_hybrid_prompt))

    return summary

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="logit_rag_stage2",
        choices=["llm", "prompt_rag", "logit_rag_stage1", "logit_rag_stage2"],
    )
    parser.add_argument("--data", default="data/datasets/qa/nq/nq_dev.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out_csv", type=str, default="dataset_eval.csv")
    parser.add_argument("--print_first_n", type=int, default=10)
    parser.add_argument("--tqdm_update_every", type=int, default=10)
    parser.add_argument("--save_to_file", action="store_true", default=False)

    # Stage-1 knobs
    parser.add_argument("--stage1_max_candidates", type=int, default=40)
    parser.add_argument("--stage1_score_top_n", type=int, default=20)
    parser.add_argument("--stage1_alpha_prior", type=float, default=0.0)
    parser.add_argument("--stage1_no_length_norm", action="store_true")

    # Stage-2 knobs
    parser.add_argument("--stage2_alpha", type=float, default=0.2)
    parser.add_argument("--stage2_alpha_sweep", type=str, default="8,10,12,15,20")  # <-- NEW 0.1,0.2,0.4,0.8
    parser.add_argument("--stage2_max_phrases", type=int, default=20)
    parser.add_argument("--stage2_hybrid_prompt", action="store_true")

    # Optional: write one summary row per sweep setting
    parser.add_argument("--sweep_out_jsonl", type=str, default="stage2_alpha_sweep.csv")  # <-- NEW

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")

    # Configure top_k + prompt_build_method based on mode
    if args.mode == "llm":
        top_k = 0
        cfg.prompt_build_method = "LLM_ONLY"
    elif args.mode == "prompt_rag":
        top_k = 5
        cfg.prompt_build_method = "QA_OPEN"
    elif args.mode == "logit_rag_stage1":
        top_k = 10
        cfg.prompt_build_method = "LOGIT_RAG_STAGE1"
    elif args.mode == "logit_rag_stage2":
        top_k = 10
        cfg.prompt_build_method = "LOGIT_RAG_STAGE2"
    else:
        raise ValueError(f"Invalid mode {args.mode}")

    out_csv_base = args.out_csv.strip()
    if out_csv_base:
        out_csv_base = f"{args.mode}_{out_csv_base}"
        logger.info("Output CSV enabled base: %s", out_csv_base)

    # Build pipeline once (reuse across sweeps)
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
        top_k=top_k,
        logger=logger,
        retrieve_only=False,
        always_log_results=False,

    )

    examples = load_jsonl(args.data)

    # Decide alphas to run
    sweep_alphas = parse_float_list(args.stage2_alpha_sweep) if args.mode == "logit_rag_stage2" else []
    
    if not sweep_alphas:
        sweep_alphas = [args.stage2_alpha]  # single run fallback

    all_summaries = []

    for alpha in sweep_alphas:
        # If sweeping, make per-alpha CSV to avoid mixing settings in one file.
        out_csv = None
        if out_csv_base:
            if len(sweep_alphas) == 1:
                out_csv = out_csv_base
            else:
                out_csv = out_csv_base.replace(".csv", f"_alpha{alpha:g}.csv")

        logger.info("Running mode=%s alpha=%s max_phrases=%s hybrid_prompt=%s limit=%s",
                    args.mode, alpha, args.stage2_max_phrases, args.stage2_hybrid_prompt, args.limit)

        metrics = evaluate(
            pipeline,
            examples,
            mode=args.mode,
            limit=(args.limit if args.limit > 0 else None),
            print_first_n=args.print_first_n,
            out_csv=(out_csv or None),
            tqdm_update_every=args.tqdm_update_every,
            stage1_max_candidates=args.stage1_max_candidates,
            stage1_score_top_n=args.stage1_score_top_n,
            stage1_length_normalize=(not args.stage1_no_length_norm),
            stage1_alpha_prior=args.stage1_alpha_prior,
            stage2_alpha=float(alpha),
            stage2_hybrid_prompt=bool(args.stage2_hybrid_prompt),
            stage2_max_phrases=int(args.stage2_max_phrases),
            save_to_file=args.save_to_file,
        )

        # annotate summary with sweep params
        metrics = dict(metrics)
        metrics["sweep_alpha"] = float(alpha)
        metrics["sweep_max_phrases"] = int(args.stage2_max_phrases)
        metrics["sweep_hybrid_prompt"] = int(bool(args.stage2_hybrid_prompt))
        all_summaries.append(metrics)

        print("\n==== FINAL (alpha={:g}) ====".format(alpha))
        print(json.dumps(metrics, indent=2))

    # Optional: write CSV summaries for quick plotting
    if args.sweep_out_jsonl.strip():
        path = args.sweep_out_jsonl.strip()
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            # Write header once
            if not file_exists:
                writer.writeheader()
            for row in all_summaries:
                writer.writerow(row)
        logger.info("Wrote sweep summaries to %s", path)


if __name__ == "__main__":
    main()