# evaluate_dataset.py (second pass: cleaner loop, unified row schema, typed results/diags)
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
from tqdm.auto import tqdm

from faasrag.core.args import RagServiceConfig
from faasrag.core.utils import (
    append_csv_row,
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    normalize_answer,
    parse_float_list,
)
from faasrag.core.rag_pipeline import RagPipeline


# =============================================================================
# IO helpers
# =============================================================================
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


# =============================================================================
# Small coercion helpers
# =============================================================================
def _f(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except Exception:
        return float(default)


def _i(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default) or default)
    except Exception:
        return int(default)


# =============================================================================
# Mode / config
# =============================================================================
VALID_MODES = {"llm", "prompt_rag", "logit_rag_stage1", "logit_rag_stage2"}


def validate_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode}. Choose: {', '.join(sorted(VALID_MODES))}")
    return mode


@dataclass(frozen=True)
class RunConfig:
    mode: str
    k: int

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
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


# =============================================================================
# Unified schemas (big win: stable CSV columns across modes)
# =============================================================================
RESULT_COLUMNS: List[str] = [
    # run identity + config
    "run_id",
    "mode",
    "stage1_max_candidates",
    "stage1_score_top_n",
    "stage1_length_normalize",
    "stage1_alpha_prior",
    "stage2_alpha",
    "stage2_hybrid_prompt",
    "stage2_max_candidates",
    "stage2_phrase_score_temperature",
    "stage2_per_token_cap",
    "stage2_clamp_first_line",
    "stage2_max_bias_steps",
    # example
    "id",
    "question",
    "prediction",
    "golden_answers",
    # metrics
    "em",
    "f1",
    # retrieval
    "top_k",
    "retrieved_count",
    "retrieved_doc_ids",
    # tokens
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    # timing
    "embed_s",
    "ann_s",
    "docstore_s",
    "prompt_build_s",
    "decode_s",
    "total_s",
    "candidate_mine_s",
    "candidate_score_s",
    # cache
    "cache_used",
    "cache_hits",
    "cache_misses",
    # stage1 diagnostics (always present, even if blank)
    "gold_in_candidates",
    "selected_gold_given_present",
    "best_candidate",
    "num_candidates_scored",
    "mean_tokens_per_candidate",
    # stage2 diagnostics (always present, even if blank)
    "bias_tokens",
    "mined_hit",
    "gold_in_mined",
    "oracle_em_from_mined",
    "selected_gold_given_gold_in_mined",
]

SUMMARY_COLUMNS: List[str] = [
    "run_id",
    "mode",
    "n",
    "em",
    "f1",
    "mean_total_tokens",
    "mean_embed_s",
    "mean_ann_s",
    "mean_docstore_s",
    "mean_prompt_build_s",
    "mean_decode_s",
    "mean_total_s",
    "mean_candidate_mine_s",
    "mean_candidate_score_s",
    # stage1 summary
    "candidate_recall",
    "selection_accuracy_given_present",
    "mean_num_candidates_scored",
    "mean_tokens_per_candidate",
    # stage2 summary
    "mined_hit_rate",
    "gold_in_mined_rate",
    "oracle_em_from_mined_rate",
    "selection_accuracy_given_gold_in_mined",
    "selection_total_where_gold_in_mined",
    # cfg dump (for convenience / joins)
    # (added dynamically as cfg_* fields)
]


def _row_with_schema(cols: List[str], values: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every column exists (stable CSV), fill missing with None."""
    return {c: values.get(c, None) for c in cols}


# =============================================================================
# Output parsing + stage diagnostics
# =============================================================================
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


@dataclass
class Stage1Diag:
    gold_in_candidates: bool = False
    selected_gold_given_present: bool = False
    best_candidate: str = ""
    num_candidates_scored: int = 0
    mean_tokens_per_candidate: float = 0.0


@dataclass
class Stage2Diag:
    bias_tokens: int = 0
    mined_hit: bool = False
    gold_in_mined: bool = False
    oracle_em_from_mined: bool = False
    selected_gold_given_gold_in_mined: bool = False


@dataclass
class ModelOut:
    pred: str
    timings: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieved_doc_ids: List[Any]
    cache_used: bool
    cache_hits: int
    cache_misses: int
    candidates: List[Dict[str, Any]]
    best_candidate: str
    mined_candidates: List[Any]
    bias_tokens: int


def parse_model_out(out: Dict[str, Any]) -> ModelOut:
    extra = out.get("extra") or {}
    timings = out.get("timings_s") or {}

    pred = (out.get("raw_answer") or out.get("answer") or "").strip()

    candidates = out.get("candidates") or extra.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []

    best = out.get("best_candidate") or extra.get("best_candidate") or ""

    cache_used = bool(out.get("cache_used", extra.get("cache_used", False)))
    cache_hits = int((out.get("cache_hits", extra.get("cache_hits", 0)) or 0))
    cache_misses = int((out.get("cache_misses", extra.get("cache_misses", 0)) or 0))

    mined_candidates = out.get("mined_candidates") or []
    bias_tokens = _i(out, "bias_tokens", 0)

    return ModelOut(
        pred=pred,
        timings=timings,
        prompt_tokens=_i(out, "prompt_tokens", 0),
        completion_tokens=_i(out, "completion_tokens", 0),
        total_tokens=_i(out, "total_tokens", 0),
        retrieved_doc_ids=out.get("retrieved_doc_ids") or [],
        cache_used=cache_used,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        candidates=candidates,
        best_candidate=str(best or ""),
        mined_candidates=mined_candidates,
        bias_tokens=bias_tokens,
    )


def compute_stage1_diag(mo: ModelOut, golds: List[str], mode: str) -> Stage1Diag:
    if mode != "logit_rag_stage1":
        return Stage1Diag()

    num = len(mo.candidates)
    has_gold = gold_in_candidates(mo.candidates, golds)
    selected = model_selected_gold(mo.best_candidate, golds) if has_gold else False
    mean_tok = (float(mo.total_tokens) / float(num)) if num > 0 else 0.0

    return Stage1Diag(
        gold_in_candidates=has_gold,
        selected_gold_given_present=selected,
        best_candidate=mo.best_candidate,
        num_candidates_scored=num,
        mean_tokens_per_candidate=mean_tok,
    )


def compute_stage2_diag(mo: ModelOut, golds: List[str], mode: str) -> Stage2Diag:
    if mode != "logit_rag_stage2":
        return Stage2Diag()

    mined = mo.mined_candidates or []
    mined_norm = [normalize_answer(str(x)) for x in mined]
    mined_norm = [x for x in mined_norm if x]

    pred_norm = normalize_answer(mo.pred)
    mined_hit = any(m and (m in pred_norm) for m in mined_norm[:30])

    gold_norms = [normalize_answer(str(g)) for g in golds]
    gold_norms = [g for g in gold_norms if g]

    mined_set = set(mined_norm)
    gold_in_mined = any(g in mined_set for g in gold_norms)

    oracle = gold_in_mined
    selected_given_oracle = oracle and any(pred_norm == g for g in gold_norms)

    return Stage2Diag(
        bias_tokens=int(mo.bias_tokens),
        mined_hit=mined_hit,
        gold_in_mined=gold_in_mined,
        oracle_em_from_mined=oracle,
        selected_gold_given_gold_in_mined=selected_given_oracle,
    )


# =============================================================================
# Runner (single dispatch point)
# =============================================================================
def run_query(pipeline: RagPipeline, q: str, cfg: RunConfig) -> Dict[str, Any]:
    mode = validate_mode(cfg.mode)

    if mode == "logit_rag_stage1":
        return pipeline.run_logit_rag_stage1(
            q,
            max_candidates=cfg.stage1_max_candidates,
            score_top_n=cfg.stage1_score_top_n,
            length_normalize=cfg.stage1_length_normalize,
            alpha_prior=cfg.stage1_alpha_prior,
        )

    if mode == "logit_rag_stage2":
        return pipeline.run_logit_rag_stage2(
            q,
            max_candidates=cfg.stage2_max_candidates,
            alpha=cfg.stage2_alpha,
            phrase_score_temperature=cfg.stage2_phrase_score_temperature,
            per_token_cap=cfg.stage2_per_token_cap,
            clamp_first_line=cfg.stage2_clamp_first_line,
            hybrid_prompt=cfg.stage2_hybrid_prompt,
            max_bias_steps=cfg.stage2_max_bias_steps,
        )

    if mode in {"prompt_rag", "llm"}:
        # If you truly want LLM-only, consider adding pipeline.run_llm_only(q)
        return pipeline.run_prompt_rag(q)

    raise RuntimeError("unreachable")


# =============================================================================
# Evaluation result (per example)
# =============================================================================
@dataclass
class ExampleResult:
    ex_id: str
    question: str
    messages: List[str]
    golds: List[str]

    prediction: str
    em: float
    f1: float

    # tokens
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # timings
    embed_s: float
    ann_s: float
    docstore_s: float
    prompt_build_s: float
    decode_s: float
    total_s: float
    candidate_mine_s: float
    candidate_score_s: float

    # retrieval
    top_k: int
    retrieved_doc_ids: List[Any]

    # cache
    cache_used: bool
    cache_hits: int
    cache_misses: int

    # diags
    stage1: Stage1Diag
    stage2: Stage2Diag


def evaluate_example(
    *,
    pipeline: RagPipeline,
    ex: Dict[str, Any],
    cfg: RunConfig,
) -> Optional[ExampleResult]:
    ex_id = ex.get("id", "")
    q, golds = get_question_and_answers(ex)
    if not q:
        return None


    out = run_query(pipeline, q, cfg)
    mo = parse_model_out(out)
    messages =out.get("messages") or []  
    em = float(metric_max_over_ground_truths(exact_match_score, mo.pred, golds))
    f1 = float(metric_max_over_ground_truths(f1_score, mo.pred, golds))

    tim = mo.timings
    embed_s = _f(tim, "embed_s")
    ann_s = _f(tim, "ann_s")
    docstore_s = _f(tim, "docstore_s")
    prompt_s = _f(tim, "prompt_s")
    decode_s = _f(tim, "decode_s")
    total_s = _f(tim, "total_s")
    mine_s = _f(tim, "candidate_mine_s")
    cand_score_s = _f(tim, "candidate_score_s")

    mode = validate_mode(cfg.mode)
    s1 = compute_stage1_diag(mo, golds, mode)
    s2 = compute_stage2_diag(mo, golds, mode)

    return ExampleResult(
        ex_id=str(ex_id),
        question=q,
        messages=messages,  # TODO if we want to save messages, need to add to pipeline output
        golds=golds,
        prediction=mo.pred,
        em=em,
        f1=f1,
        prompt_tokens=int(mo.prompt_tokens),
        completion_tokens=int(mo.completion_tokens),
        total_tokens=int(mo.total_tokens),
        embed_s=embed_s,
        ann_s=ann_s,
        docstore_s=docstore_s,
        prompt_build_s=prompt_s,
        decode_s=decode_s,
        total_s=total_s,
        candidate_mine_s=mine_s,
        candidate_score_s=cand_score_s,
        top_k=int(getattr(pipeline, "top_k", 0)),
        retrieved_doc_ids=mo.retrieved_doc_ids,
        cache_used=bool(mo.cache_used),
        cache_hits=int(mo.cache_hits),
        cache_misses=int(mo.cache_misses),
        stage1=s1,
        stage2=s2,
    )


# =============================================================================
# Aggregator (keeps loop tiny)
# =============================================================================
@dataclass
class Aggregator:
    n: int = 0
    em_sum: float = 0.0
    f1_sum: float = 0.0

    total_tok_sum: int = 0
    embed_sum: float = 0.0
    ann_sum: float = 0.0
    docstore_sum: float = 0.0
    prompt_build_sum: float = 0.0
    decode_sum: float = 0.0
    total_s_sum: float = 0.0
    mine_sum: float = 0.0
    score_sum: float = 0.0

    # stage1
    cand_recall_hits: int = 0
    cand_recall_total: int = 0
    select_hits: int = 0
    select_total: int = 0
    stage1_num_scored_sum: int = 0
    stage1_tok_per_cand_sum: float = 0.0
    stage1_tok_per_cand_n: int = 0

    # stage2
    stage2_n: int = 0
    stage2_mined_hit_sum: int = 0
    stage2_gold_in_mined_sum: int = 0
    stage2_oracle_hits: int = 0
    stage2_oracle_total: int = 0
    stage2_select_hits_given_oracle: int = 0
    stage2_select_total_given_oracle: int = 0

    def add(self, res: ExampleResult, mode: str) -> None:
        self.n += 1
        self.em_sum += res.em
        self.f1_sum += res.f1

        self.total_tok_sum += res.total_tokens
        self.embed_sum += res.embed_s
        self.ann_sum += res.ann_s
        self.docstore_sum += res.docstore_s
        self.prompt_build_sum += res.prompt_build_s
        self.decode_sum += res.decode_s
        self.total_s_sum += res.total_s
        self.mine_sum += res.candidate_mine_s
        self.score_sum += res.candidate_score_s

        if mode == "logit_rag_stage1":
            self.cand_recall_total += 1
            self.cand_recall_hits += int(res.stage1.gold_in_candidates)
            if res.stage1.gold_in_candidates:
                self.select_total += 1
                self.select_hits += int(res.stage1.selected_gold_given_present)

            self.stage1_num_scored_sum += int(res.stage1.num_candidates_scored)
            if res.stage1.num_candidates_scored > 0:
                self.stage1_tok_per_cand_sum += float(res.stage1.mean_tokens_per_candidate)
                self.stage1_tok_per_cand_n += 1

        if mode == "logit_rag_stage2":
            self.stage2_n += 1
            self.stage2_mined_hit_sum += int(res.stage2.mined_hit)
            self.stage2_gold_in_mined_sum += int(res.stage2.gold_in_mined)
            self.stage2_oracle_total += 1
            self.stage2_oracle_hits += int(res.stage2.oracle_em_from_mined)

            if res.stage2.oracle_em_from_mined:
                self.stage2_select_total_given_oracle += 1
                self.stage2_select_hits_given_oracle += int(res.stage2.selected_gold_given_gold_in_mined)

    def mean(self, x: float) -> float:
        return x / self.n if self.n else 0.0

    def mean_int(self, x: int) -> float:
        return float(x) / self.n if self.n else 0.0

    def postfix(self, mode: str) -> Dict[str, str]:
        if self.n == 0:
            return {}
        post = {
            "EM": f"{self.em_sum/self.n:.3f}",
            "F1": f"{self.f1_sum/self.n:.3f}",
            "tok": f"{(self.total_tok_sum/self.n):.0f}",
            "tot_s": f"{(self.total_s_sum/self.n):.2f}",
        }
        if mode == "logit_rag_stage1":
            cand_rec = (self.cand_recall_hits / self.cand_recall_total) if self.cand_recall_total else 0.0
            sel_acc = (self.select_hits / self.select_total) if self.select_total else 0.0
            post["cand@"] = f"{cand_rec:.2f}"
            post["sel@"] = f"{sel_acc:.2f}"
            if self.stage1_tok_per_cand_n:
                post["tok/cand"] = f"{(self.stage1_tok_per_cand_sum / self.stage1_tok_per_cand_n):.1f}"
        if mode == "logit_rag_stage2" and self.stage2_n:
            post["mined_hit"] = f"{(self.stage2_mined_hit_sum/self.stage2_n):.2f}"
            post["gold_in_m"] = f"{(self.stage2_gold_in_mined_sum/self.stage2_n):.2f}"
            post["oracle"] = f"{(self.stage2_oracle_hits/self.stage2_oracle_total):.2f}" if self.stage2_oracle_total else "0.00"
            post["sel|oracle"] = (
                f"{(self.stage2_select_hits_given_oracle/self.stage2_select_total_given_oracle):.2f}"
                if self.stage2_select_total_given_oracle else "0.00"
            )
        return post


# =============================================================================
# Row building (single place, stable schema)
# =============================================================================
def build_result_row(
    *,
    run_id: str,
    cfg: RunConfig,
    pipeline: RagPipeline,
    res: ExampleResult,
) -> Dict[str, Any]:
    mode = validate_mode(cfg.mode)

    values: Dict[str, Any] = {
        # run identity + config
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
        "id": res.ex_id,
        "question": res.question,
        "prediction": res.prediction,
        "golden_answers": json.dumps(res.golds, ensure_ascii=False),
        # metrics
        "em": float(res.em),
        "f1": float(res.f1),
        # retrieval
        "top_k": int(getattr(pipeline, "top_k", 0)),
        "retrieved_count": len(res.retrieved_doc_ids),
        "retrieved_doc_ids": json.dumps(res.retrieved_doc_ids),
        # tokens
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        # timing
        "embed_s": res.embed_s,
        "ann_s": res.ann_s,
        "docstore_s": res.docstore_s,
        "prompt_build_s": res.prompt_build_s,
        "decode_s": res.decode_s,
        "total_s": res.total_s,
        "candidate_mine_s": res.candidate_mine_s,
        "candidate_score_s": res.candidate_score_s,
        # cache
        "cache_used": int(bool(res.cache_used)),
        "cache_hits": int(res.cache_hits),
        "cache_misses": int(res.cache_misses),
        # stage1 (always present)
        "gold_in_candidates": int(res.stage1.gold_in_candidates),
        "selected_gold_given_present": int(res.stage1.selected_gold_given_present),
        "best_candidate": res.stage1.best_candidate,
        "num_candidates_scored": int(res.stage1.num_candidates_scored),
        "mean_tokens_per_candidate": float(res.stage1.mean_tokens_per_candidate),
        # stage2 (always present)
        "bias_tokens": int(res.stage2.bias_tokens),
        "mined_hit": int(res.stage2.mined_hit),
        "gold_in_mined": int(res.stage2.gold_in_mined),
        "oracle_em_from_mined": int(res.stage2.oracle_em_from_mined),
        "selected_gold_given_gold_in_mined": int(res.stage2.selected_gold_given_gold_in_mined),
    }

    return _row_with_schema(RESULT_COLUMNS, values)


# =============================================================================
# Core evaluation (now orchestration-only)
# =============================================================================
def evaluate_one_run(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    *,
    cfg: RunConfig,
    results_csv: Optional[str],
    save_to_file: bool = True,
) -> Dict[str, Any]:
    
    mode = validate_mode(cfg.mode)
    run_id = cfg.run_id()
    agg = Aggregator()
    subset = examples[: cfg.limit] if cfg.limit else examples
    pbar = tqdm(subset, total=len(subset), desc=f"Evaluating ({mode})", dynamic_ncols=True)

    for ex in pbar:
        res = evaluate_example(pipeline=pipeline, ex=ex, cfg=cfg)
        if res is None:
            continue

        agg.add(res, mode)

        if agg.n <= cfg.print_first_n:
            tqdm.write("\n---")
            tqdm.write(f"run_id: {run_id}  mode: {mode}  id: {res.ex_id}")
            tqdm.write(f"Q: {res.question}")
            tqdm.write(f"PRED: {res.prediction}")
            tqdm.write(f"GOLDS: {res.golds}")
            tqdm.write(f"EM={res.em} F1={res.f1:.3f}")
            tqdm.write(f"tokens: prompt={res.prompt_tokens} completion={res.completion_tokens} total={res.total_tokens}")
            tqdm.write(f"timings_s: total={res.total_s:.3f} decode={res.decode_s:.3f}")
            if hasattr(res, "messages") and res.messages:
                tqdm.write(f"messages: {res.messages:10}")

        if cfg.tqdm_update_every > 0 and agg.n % cfg.tqdm_update_every == 0:
            pbar.set_postfix(agg.postfix(mode))

        if save_to_file and results_csv:
            row = build_result_row(run_id=run_id, cfg=cfg, pipeline=pipeline, res=res)
            # uses your faasrag.core.utils.append_csv_row; stable schema means consistent columns
            append_csv_row(results_csv, row)

    # summary (stable-ish keys)
    summary_values: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "top_k": int(getattr(pipeline, "top_k", 0)),
        "n": agg.n,
        "em": agg.mean(agg.em_sum),
        "f1": agg.mean(agg.f1_sum),
        "mean_total_tokens": agg.mean_int(agg.total_tok_sum),
        "mean_embed_s": agg.mean(agg.embed_sum),
        "mean_ann_s": agg.mean(agg.ann_sum),
        "mean_docstore_s": agg.mean(agg.docstore_sum),
        "mean_prompt_build_s": agg.mean(agg.prompt_build_sum),
        "mean_decode_s": agg.mean(agg.decode_sum),
        "mean_total_s": agg.mean(agg.total_s_sum),
        "mean_candidate_mine_s": agg.mean(agg.mine_sum),
        "mean_candidate_score_s": agg.mean(agg.score_sum),
        # stage1
        "candidate_recall": (agg.cand_recall_hits / agg.cand_recall_total) if agg.cand_recall_total else 0.0,
        "selection_accuracy_given_present": (agg.select_hits / agg.select_total) if agg.select_total else 0.0,
        "mean_num_candidates_scored": (agg.stage1_num_scored_sum / agg.n) if agg.n else 0.0,
        "mean_tokens_per_candidate": (agg.stage1_tok_per_cand_sum / agg.stage1_tok_per_cand_n) if agg.stage1_tok_per_cand_n else 0.0,
        # stage2
        "mined_hit_rate": (agg.stage2_mined_hit_sum / agg.stage2_n) if agg.stage2_n else 0.0,
        "gold_in_mined_rate": (agg.stage2_gold_in_mined_sum / agg.stage2_n) if agg.stage2_n else 0.0,
        "oracle_em_from_mined_rate": (agg.stage2_oracle_hits / agg.stage2_oracle_total) if agg.stage2_oracle_total else 0.0,
        "selection_accuracy_given_gold_in_mined": (
            (agg.stage2_select_hits_given_oracle / agg.stage2_select_total_given_oracle)
            if agg.stage2_select_total_given_oracle else 0.0
        ),
        "selection_total_where_gold_in_mined": agg.stage2_select_total_given_oracle,
    }

    # add cfg_* fields
    summary_values.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})

    # (optional) enforce a stable subset of columns
    # keep all cfg_* fields, so we won't hard-truncate.
    return summary_values


# =============================================================================
# CLI + Hydra entry
# =============================================================================
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="prompt_rag", choices=sorted(VALID_MODES))
    parser.add_argument("--data", default="data/datasets/qa/nq/nq_train.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--print_first_n", type=int, default=10)
    parser.add_argument("--tqdm_update_every", type=int, default=10)

    parser.add_argument("--results_csv", type=str, default="all_results.csv")
    parser.add_argument("--summary_csv", type=str, default="run_summaries.csv")
    parser.add_argument("--save_to_file", action="store_true", default=True)

    # Stage-1 knobs
    parser.add_argument("--stage1_max_candidates", type=int, default=40)
    parser.add_argument("--stage1_score_top_n", type=int, default=20)
    parser.add_argument("--stage1_alpha_prior", type=float, default=0.0)
    parser.add_argument("--stage1_no_length_norm", action="store_true")

    # Stage-2 knobs + sweeps
    parser.add_argument("--stage2_alpha", type=float, default=10)
    parser.add_argument("--stage2_alpha_sweep", type=str, default="")
    parser.add_argument("--stage2_max_candidates", type=int, default=2)
    parser.add_argument("--stage2_max_candidates_sweep", type=str, default="")
    parser.add_argument("--stage2_phrase_score_temperature", type=float, default=0.5)
    parser.add_argument("--stage2_phrase_temp_sweep", type=str, default="")
    parser.add_argument("--stage2_per_token_cap", type=float, default=1.5)
    parser.add_argument("--stage2_per_token_cap_sweep", type=str, default="")
    parser.add_argument("--stage2_clamp_first_line", action="store_true")
    parser.add_argument("--stage2_hybrid_prompt", action="store_true")
    parser.add_argument("--stage2_max_bias_steps", type=int, default=None)

    parser.add_argument("--sweep_style", type=str, default="grid", choices=["grid", "zip"])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")

    mode = validate_mode(args.mode)

    # Configure pipeline based on mode
    if mode == "llm":
        top_k = 0
        cfg.prompt_build_method = "LLM_ONLY"
    elif mode == "prompt_rag":
        top_k = 10
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

    # sweeps
    def sweep_list(s: str, default_val: float) -> List[float]:
        vals = parse_float_list(s) if s.strip() else []
        return vals if vals else [default_val]

    alpha_list = sweep_list(args.stage2_alpha_sweep, args.stage2_alpha)
    maxcand_list = [int(x) for x in sweep_list(args.stage2_max_candidates_sweep, float(args.stage2_max_candidates))]
    temp_list = sweep_list(args.stage2_phrase_temp_sweep, args.stage2_phrase_score_temperature)
    cap_list = sweep_list(args.stage2_per_token_cap_sweep, args.stage2_per_token_cap)

    runs: List[RunConfig] = []
    if mode != "logit_rag_stage2":
        runs = [
            RunConfig(
                mode=mode,
                k=top_k,
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
            )
        ]
    else:
        if args.sweep_style == "zip":
            k = max(len(alpha_list), len(maxcand_list), len(temp_list), len(cap_list))

            def pad(xs: List[Any], k_: int) -> List[Any]:
                return xs + [xs[-1]] * (k_ - len(xs))

            alpha_list = pad(alpha_list, k)
            maxcand_list = pad(maxcand_list, k)
            temp_list = pad(temp_list, k)
            cap_list = pad(cap_list, k)
            combos = zip(alpha_list, maxcand_list, temp_list, cap_list)
        else:
            combos = ((a, mc, t, c) for a in alpha_list for mc in maxcand_list for t in temp_list for c in cap_list)

        for a, mc, t, c in combos:
            runs.append(
                RunConfig(
                    mode=mode,
                    k=top_k,
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
                )
            )

    logger.info(
        "Running %d configs (mode=%s). results_csv=%s summary_csv=%s save=%s",
        len(runs),
        mode,
        args.results_csv,
        args.summary_csv,
        args.save_to_file,
    )

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

        if args.summary_csv:
            append_csv_row(args.summary_csv, summary)
            logger.info("Appended summary row to %s (run_id=%s)", args.summary_csv, summary.get("run_id"))


if __name__ == "__main__":
    main()
