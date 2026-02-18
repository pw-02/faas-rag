# evaluate_dataset.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
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
    parse_float_list,
)
from faasrag.core.rag_pipeline import RagPipeline


# ----------------------------
# Data helpers
# ----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def append_summary_row(path: str, row: Dict[str, Any]) -> None:
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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
    return any(normalize_answer(g) in cand_norm for g in golds)


def model_selected_gold(best_candidate: str, golds: List[str]) -> bool:
    if not best_candidate or not golds:
        return False
    b = normalize_answer(best_candidate)
    return any(b == normalize_answer(g) for g in golds)


# ----------------------------
# Config objects
# ----------------------------
@dataclass(frozen=True)
class RunConfig:
    mode: str

    # general
    limit: Optional[int]
    print_first_n: int
    tqdm_update_every: int

    # stage1 knobs
    stage1_max_candidates: int
    stage1_score_top_n: int
    stage1_length_normalize: bool
    stage1_alpha_prior: float

    # stage2 knobs
    stage2_alpha: float
    stage2_hybrid_prompt: bool
    stage2_max_candidates: int
    stage2_phrase_score_temperature: float
    stage2_per_token_cap: float
    stage2_clamp_first_line: bool
    stage2_max_bias_steps: Optional[int]

    def run_id(self) -> str:
        # stable ID for joining results across files/tools
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


def validate_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    valid = {"llm", "prompt_rag", "logit_rag_stage1", "logit_rag_stage2"}
    if mode not in valid:
        raise ValueError(f"Invalid mode {mode}. Choose: {', '.join(sorted(valid))}")
    return mode


# ----------------------------
# Evaluation core
# ----------------------------
def evaluate_one_run(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    *,
    cfg: RunConfig,
    results_csv: Optional[str],
    save_to_file: bool,
) -> Dict[str, Any]:
    mode = validate_mode(cfg.mode)

    # aggregates
    n = 0
    em_sum = 0.0
    f1_sum = 0.0

    prompt_tok_sum = 0
    completion_tok_sum = 0
    total_tok_sum = 0

    total_s_sum = 0.0
    decode_sum = 0.0
    embed_sum = 0.0
    ann_sum = 0.0
    docstore_sum = 0.0
    prompt_build_sum = 0.0

    stage1_mine_sum = 0.0
    stage1_score_sum = 0.0

    stage1_num_scored_sum = 0
    stage1_tok_per_cand_sum = 0.0
    stage1_tok_per_cand_n = 0

    cand_recall_hits = 0
    cand_recall_total = 0
    select_hits = 0
    select_total = 0

    stage2_n = 0
    stage2_mined_hit_sum = 0
    stage2_gold_in_mined_sum = 0
    stage2_oracle_hits = 0
    stage2_oracle_total = 0
    stage2_select_hits_given_oracle = 0
    stage2_select_total_given_oracle = 0

    subset = examples[: cfg.limit] if cfg.limit else examples
    pbar = tqdm(subset, total=len(subset), desc=f"Evaluating ({mode})", dynamic_ncols=True)

    def mean(x: float) -> float:
        return x / n if n else 0.0

    run_id = cfg.run_id()

    for ex in pbar:
        ex_id = ex.get("id", "")
        q, golds = get_question_and_answers(ex)
        if not q:
            continue

        # Dispatch
        if mode == "logit_rag_stage1":
            out = pipeline.run_logit_rag_stage1(
                q,
                max_candidates=cfg.stage1_max_candidates,
                score_top_n=cfg.stage1_score_top_n,
                length_normalize=cfg.stage1_length_normalize,
                alpha_prior=cfg.stage1_alpha_prior,
            )
        elif mode == "logit_rag_stage2":
            out = pipeline.run_logit_rag_stage2(
                q,
                max_candidates=cfg.stage2_max_candidates,
                alpha=cfg.stage2_alpha,
                phrase_score_temperature=cfg.stage2_phrase_score_temperature,
                per_token_cap=cfg.stage2_per_token_cap,
                clamp_first_line=cfg.stage2_clamp_first_line,
                hybrid_prompt=cfg.stage2_hybrid_prompt,
                max_bias_steps=cfg.stage2_max_bias_steps,
            )
        elif mode in {"prompt_rag", "llm"}:
            # If you truly want LLM-only, consider adding pipeline.run_llm_only(q)
            out = pipeline.run_prompt_rag(q)
        else:
            raise RuntimeError("unreachable")

        pred = (out.get("raw_answer") or out.get("answer") or "").strip()
        em = metric_max_over_ground_truths(exact_match_score, pred, golds)
        f1 = metric_max_over_ground_truths(f1_score, pred, golds)

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

        extra = out.get("extra") or {}
        cache_used = out.get("cache_used", extra.get("cache_used", False))
        cache_hits = out.get("cache_hits", extra.get("cache_hits", 0))
        cache_misses = out.get("cache_misses", extra.get("cache_misses", 0))

        candidates = out.get("candidates") or extra.get("candidates") or []
        best = out.get("best_candidate") or extra.get("best_candidate") or ""

        num_candidates_scored = len(candidates) if isinstance(candidates, list) else 0

        mean_tokens_per_candidate = 0.0
        if mode == "logit_rag_stage1" and num_candidates_scored > 0:
            mean_tokens_per_candidate = float(total_tokens) / float(num_candidates_scored)
            stage1_num_scored_sum += int(num_candidates_scored)
            stage1_tok_per_cand_sum += float(mean_tokens_per_candidate)
            stage1_tok_per_cand_n += 1

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

        mined_hit = False
        gold_in_mined = False
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

            stage2_oracle_total += 1
            oracle_em_from_mined = gold_in_mined
            stage2_oracle_hits += int(oracle_em_from_mined)

            if oracle_em_from_mined:
                stage2_select_total_given_oracle += 1
                selected_gold_given_oracle = any(pred_norm == g for g in gold_norms)
                stage2_select_hits_given_oracle += int(selected_gold_given_oracle)

        # totals
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

        # debug prints
        if n <= cfg.print_first_n:
            tqdm.write("\n---")
            tqdm.write(f"run_id: {run_id}  mode: {mode}  id: {ex_id}")
            tqdm.write(f"Q: {q}")
            tqdm.write(f"PRED: {pred}")
            tqdm.write(f"GOLDS: {golds}")
            tqdm.write(f"EM={em} F1={f1:.3f}")
            tqdm.write(f"tokens: prompt={prompt_tokens} completion={completion_tokens} total={total_tokens}")
            tqdm.write(f"timings_s: total={total_s:.3f} decode={decode_s:.3f}")

        # progress postfix
        if cfg.tqdm_update_every > 0 and n % cfg.tqdm_update_every == 0:
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
                postfix["oracle"] = f"{(stage2_oracle_hits/stage2_oracle_total):.2f}" if stage2_oracle_total else "0.00"
                postfix["sel|oracle"] = (
                    f"{(stage2_select_hits_given_oracle/stage2_select_total_given_oracle):.2f}"
                    if stage2_select_total_given_oracle else "0.00"
                )
            pbar.set_postfix(postfix)

        # per-example row (unified across runs)
        row: Dict[str, Any] = {
            # run identity + config (so later you can groupby)
            "run_id": run_id,
            "mode": mode,
            "stage1_max_candidates": cfg.stage1_max_candidates,
            "stage1_score_top_n": cfg.stage1_score_top_n,
            "stage1_length_normalize": int(cfg.stage1_length_normalize),
            "stage1_alpha_prior": cfg.stage1_alpha_prior,
            "stage2_alpha": cfg.stage2_alpha,
            "stage2_hybrid_prompt": int(cfg.stage2_hybrid_prompt),
            "stage2_max_candidates": cfg.stage2_max_candidates,
            "stage2_phrase_score_temperature": cfg.stage2_phrase_score_temperature,
            "stage2_per_token_cap": cfg.stage2_per_token_cap,
            "stage2_clamp_first_line": int(cfg.stage2_clamp_first_line),
            "stage2_max_bias_steps": cfg.stage2_max_bias_steps,

            # example
            "id": ex_id,
            "question": q,
            "prediction": pred,
            "golden_answers": json.dumps(golds, ensure_ascii=False),

            # metrics
            "em": float(em),
            "f1": float(f1),

            # retrieval
            "top_k": int(getattr(pipeline, "top_k", 0)),
            "retrieved_count": len(retrieved_doc_ids),
            "retrieved_doc_ids": json.dumps(retrieved_doc_ids),

            # tokens
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,

            # timing
            "embed_s": embed_s,
            "ann_s": ann_s,
            "docstore_s": docstore_s,
            "prompt_build_s": prompt_s,
            "decode_s": decode_s,
            "total_s": total_s,

            "candidate_mine_s": mine_s,
            "candidate_score_s": cand_score_s,

            # cache
            "cache_used": int(bool(cache_used)),
            "cache_hits": int(cache_hits or 0),
            "cache_misses": int(cache_misses or 0),
        }

        if mode == "logit_rag_stage1":
            row.update({
                "gold_in_candidates": int(has_gold),
                "selected_gold_given_present": int(selected_gold),
                "best_candidate": best,
                "num_candidates_scored": int(num_candidates_scored),
                "mean_tokens_per_candidate": float(mean_tokens_per_candidate),
            })

        if mode == "logit_rag_stage2":
            row.update({
                "bias_tokens": int(out.get("bias_tokens", 0) or 0),
                "mined_hit": int(mined_hit),
                "gold_in_mined": int(gold_in_mined),
                "oracle_em_from_mined": int(oracle_em_from_mined),
                "selected_gold_given_gold_in_mined": int(selected_gold_given_oracle),
            })

        if save_to_file and results_csv:
            append_csv_row(results_csv, row)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "n": n,
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
        summary.update({
            "candidate_recall": (cand_recall_hits / cand_recall_total) if cand_recall_total else 0.0,
            "selection_accuracy_given_present": (select_hits / select_total) if select_total else 0.0,
            "mean_num_candidates_scored": (stage1_num_scored_sum / n) if n else 0.0,
            "mean_tokens_per_candidate": (
                (stage1_tok_per_cand_sum / stage1_tok_per_cand_n) if stage1_tok_per_cand_n else 0.0
            ),
        })

    if mode == "logit_rag_stage2":
        summary.update({
            "mined_hit_rate": (stage2_mined_hit_sum / stage2_n) if stage2_n else 0.0,
            "gold_in_mined_rate": (stage2_gold_in_mined_sum / stage2_n) if stage2_n else 0.0,
            "oracle_em_from_mined_rate": (stage2_oracle_hits / stage2_oracle_total) if stage2_oracle_total else 0.0,
            "selection_accuracy_given_gold_in_mined": (
                (stage2_select_hits_given_oracle / stage2_select_total_given_oracle)
                if stage2_select_total_given_oracle else 0.0
            ),
            "selection_total_where_gold_in_mined": stage2_select_total_given_oracle,
        })

    # include the config in the summary row too
    summary.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
    return summary


# ----------------------------
# CLI + Hydra entry
# ----------------------------
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="llm",
                        choices=["llm", "prompt_rag", "logit_rag_stage1", "logit_rag_stage2"])
    parser.add_argument("--data", default="data/datasets/qa/nq/nq_train.jsonl")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--print_first_n", type=int, default=10)
    parser.add_argument("--tqdm_update_every", type=int, default=10)

    # unified outputs
    parser.add_argument("--results_csv", type=str, default="all_results.csv",
                        help="One big CSV with per-example rows across all sweeps.")
    parser.add_argument("--summary_csv", type=str, default="run_summaries.csv",
                        help="One row per run (config) for quick comparison.")
    parser.add_argument("--save_to_file", action="store_true", default=True)

    # Stage-1 knobs
    parser.add_argument("--stage1_max_candidates", type=int, default=40)
    parser.add_argument("--stage1_score_top_n", type=int, default=20)
    parser.add_argument("--stage1_alpha_prior", type=float, default=0.0)
    parser.add_argument("--stage1_no_length_norm", action="store_true")

    # Stage-2 knobs + sweeps (allow sweeps for multiple params)
    parser.add_argument("--stage2_alpha", type=float, default=10)
    parser.add_argument("--stage2_alpha_sweep", type=str, default="1,4,8,10,12,16,20,30,40")  # e.g. "0.1,0.2,0.4,0.8"
    parser.add_argument("--stage2_max_candidates", type=int, default=15)
    parser.add_argument("--stage2_max_candidates_sweep", type=str, default="5,10,15,20,25,30,35,40")  # e.g. "10,15,20"
    parser.add_argument("--stage2_phrase_score_temperature", type=float, default=1.0)
    parser.add_argument("--stage2_phrase_temp_sweep", type=str, default="0.1,0.5,1.0,2.0,5.0")  # e.g. "0.1,0.5,1.0,2.0"
    parser.add_argument("--stage2_per_token_cap", type=float, default=2.0)
    parser.add_argument("--stage2_per_token_cap_sweep", type=str, default="1,2,5,10")  # e.g. "1,2,5,10"
    parser.add_argument("--stage2_clamp_first_line", action="store_true")
    parser.add_argument("--stage2_hybrid_prompt", action="store_true")
    parser.add_argument("--stage2_max_bias_steps", type=int, default=None)

    # sweep style: grid or zip
    parser.add_argument("--sweep_style", type=str, default="grid", choices=["grid", "zip"])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")

    # Configure pipeline based on mode
    mode = validate_mode(args.mode)
    if mode == "llm":
        top_k = 0
        cfg.prompt_build_method = "LLM_ONLY"
    elif mode == "prompt_rag":
        top_k = 5
        cfg.prompt_build_method = "QA_OPEN"
    elif mode == "logit_rag_stage1":
        top_k = 10
        cfg.prompt_build_method = "LOGIT_RAG_STAGE1"
    else:
        top_k = 10
        cfg.prompt_build_method = "LOGIT_RAG_STAGE2"

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

    # Build sweep lists
    def sweep_list(s: str, default_val: float) -> List[float]:
        vals = parse_float_list(s) if s.strip() else []
        return vals if vals else [default_val]

    alpha_list = sweep_list(args.stage2_alpha_sweep, args.stage2_alpha)
    maxcand_list = [int(x) for x in sweep_list(args.stage2_max_candidates_sweep, float(args.stage2_max_candidates))]
    temp_list = sweep_list(args.stage2_phrase_temp_sweep, args.stage2_phrase_score_temperature)
    cap_list = sweep_list(args.stage2_per_token_cap_sweep, args.stage2_per_token_cap)

    # Create runs (grid or zip)
    runs: List[RunConfig] = []
    if mode != "logit_rag_stage2":
        runs = [RunConfig(
            mode=mode,
            limit=(args.limit if args.limit > 0 else None),
            print_first_n=args.print_first_n,
            tqdm_update_every=args.tqdm_update_every,

            stage1_max_candidates=args.stage1_max_candidates,
            stage1_score_top_n=args.stage1_score_top_n,
            stage1_length_normalize=(not args.stage1_no_length_norm),
            stage1_alpha_prior=args.stage1_alpha_prior,

            stage2_alpha=args.stage2_alpha,
            stage2_hybrid_prompt=bool(args.stage2_hybrid_prompt),
            stage2_max_candidates=args.stage2_max_candidates,
            stage2_phrase_score_temperature=args.stage2_phrase_score_temperature,
            stage2_per_token_cap=args.stage2_per_token_cap,
            stage2_clamp_first_line=bool(args.stage2_clamp_first_line),
            stage2_max_bias_steps=args.stage2_max_bias_steps,
        )]
    else:
        if args.sweep_style == "zip":
            k = max(len(alpha_list), len(maxcand_list), len(temp_list), len(cap_list))
            # pad with last element
            def pad(xs, k): return xs + [xs[-1]] * (k - len(xs))
            alpha_list = pad(alpha_list, k)
            maxcand_list = pad(maxcand_list, k)
            temp_list = pad(temp_list, k)
            cap_list = pad(cap_list, k)
            combos = zip(alpha_list, maxcand_list, temp_list, cap_list)
        else:
            # grid
            combos = ((a, mc, t, c) for a in alpha_list for mc in maxcand_list for t in temp_list for c in cap_list)

        for a, mc, t, c in combos:
            runs.append(RunConfig(
                mode=mode,
                limit=(args.limit if args.limit > 0 else None),
                print_first_n=args.print_first_n,
                tqdm_update_every=args.tqdm_update_every,

                stage1_max_candidates=args.stage1_max_candidates,
                stage1_score_top_n=args.stage1_score_top_n,
                stage1_length_normalize=(not args.stage1_no_length_norm),
                stage1_alpha_prior=args.stage1_alpha_prior,

                stage2_alpha=float(a),
                stage2_hybrid_prompt=bool(args.stage2_hybrid_prompt),
                stage2_max_candidates=int(mc),
                stage2_phrase_score_temperature=float(t),
                stage2_per_token_cap=float(c),
                stage2_clamp_first_line=bool(args.stage2_clamp_first_line),
                stage2_max_bias_steps=args.stage2_max_bias_steps,
            ))

    # If saving, clear output files once at start
    if args.save_to_file:
        if args.results_csv:
            open(args.results_csv, "w", encoding="utf-8").close()
        if args.summary_csv:
            open(args.summary_csv, "w", encoding="utf-8").close()

    logger.info("Running %d configs (mode=%s). results_csv=%s summary_csv=%s save=%s",
                len(runs), mode, args.results_csv, args.summary_csv, args.save_to_file)
    
    logger.info("Total runs to execute: %d", len(runs))
    for rc in runs:
        logger.info("RUN %s cfg=%s", rc.run_id(), json.dumps(asdict(rc), default=str))

        summary = evaluate_one_run(
            pipeline,
            examples,
            cfg=rc,
            results_csv=args.results_csv,
            save_to_file=args.save_to_file,
        )

        print("\n==== FINAL (run_id={}) ====".format(rc.run_id()))
        print(json.dumps(summary, indent=2, default=str))

        # ✅ write immediately (one row per run)
        if args.save_to_file and args.summary_csv:
            append_summary_row(args.summary_csv, summary)
            logger.info("Appended summary row to %s (run_id=%s)", args.summary_csv, summary.get("run_id"))

if __name__ == "__main__":
    main()
