# evaluate_dataset.py (fourth pass: stage1 removed, stage2 -> logit_rag, single-flag sweeps,
# prefixed cfg fields, robust mined-candidate parsing, optional debug JSONL, top-10 console preview)

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

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


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


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


def _coerce_mined_candidates(x: Any) -> List[Tuple[str, float]]:
    """
    Accepts:
      - [("phrase", score), ...]
      - [{"candidate": "...", "score": ...}, ...]
      - ["phrase", ...]  (score becomes 0.0)
    Returns: [(phrase, score), ...] with non-empty phrases only.
    """
    if not x:
        return []
    out: List[Tuple[str, float]] = []
    if not isinstance(x, list):
        return out

    for item in x:
        if isinstance(item, dict):
            phrase = str(item.get("candidate") or item.get("phrase") or "").strip()
            score_val = item.get("score", item.get("llm_score", item.get("prior", 0.0)))
            try:
                score = float(score_val) if score_val is not None else 0.0
            except Exception:
                score = 0.0
            if phrase:
                out.append((phrase, score))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            phrase = str(item[0]).strip()
            try:
                score = float(item[1]) if item[1] is not None else 0.0
            except Exception:
                score = 0.0
            if phrase:
                out.append((phrase, score))
        else:
            phrase = str(item).strip()
            if phrase:
                out.append((phrase, 0.0))
    return out


# =============================================================================
# Modes / config
# =============================================================================
VALID_MODES = {"llm", "prompt_rag", "logit_rag"}


def validate_mode(mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode}. Choose: {', '.join(sorted(VALID_MODES))}")
    return mode


@dataclass(frozen=True)
class RunConfig:
    mode: str
    top_k: int
    dataset: str

    # general
    limit: Optional[int]
    print_first_n: int
    tqdm_update_every: int

    # logit-only (None for other modes)
    logit_max_mined_candidates: Optional[int] = None
    logit_bias_strength: Optional[float] = None
    logit_max_token_logit_bias: Optional[float] = None
    logit_phrase_softmax_temperature: Optional[float] = None
    logit_clamp_first_line: Optional[bool] = None
    logit_hybrid_prompt: Optional[bool] = None
    logit_max_bias_steps: Optional[int] = None
    logit_bias_top_n: Optional[int] = None

    def run_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


# =============================================================================
# Stable CSV schemas (prefixed cfg fields + mined top10)
# =============================================================================
RESULT_COLUMNS: List[str] = [
    # run identity
    "run_id",
    "mode",
    # cfg (prefixed)
    "cfg_top_k",
    "cfg_logit_max_mined_candidates",
    "cfg_logit_bias_strength",
    "cfg_logit_max_token_logit_bias",
    "cfg_logit_phrase_softmax_temperature",
    "cfg_logit_clamp_first_line",
    "cfg_logit_hybrid_prompt",
    "cfg_logit_max_bias_steps",
    # example
    "id",
    "question",
    "prediction",
    "golden_answers",
    # metrics
    "em",
    "f1",
    # retrieval
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
    # cache
    "cache_used",
    "cache_hits",
    "cache_misses",
    # logit diagnostics
    "num_biased_token_ids",
    "mined_hit",
    "gold_in_mined",
    "oracle_em_from_mined",
    "selected_gold_given_gold_in_mined",
    # debugging
    "mined_candidates_count",
    "mined_candidates_top10_json",
    "top_biased_tokens_json",
]

SUMMARY_COLUMNS: List[str] = [
    "run_id",
    "mode",
    # cfg (prefixed)
    "cfg_top_k",
    "cfg_logit_max_mined_candidates",
    "cfg_logit_bias_strength",
    "cfg_logit_max_token_logit_bias",
    "cfg_logit_phrase_softmax_temperature",
    "cfg_logit_clamp_first_line",
    "cfg_logit_hybrid_prompt",
    "cfg_logit_max_bias_steps",
    # aggregate metrics
    "n",
    "em",
    "f1",
    "mean_total_tokens",
    # timing means
    "mean_embed_s",
    "mean_ann_s",
    "mean_docstore_s",
    "mean_prompt_build_s",
    "mean_decode_s",
    "mean_total_s",
    "mean_candidate_mine_s",
    # logit summaries
    "logit_mined_hit_rate",
    "logit_gold_in_mined_rate",
    "logit_oracle_em_from_mined_rate",
    "logit_selection_accuracy_given_gold_in_mined",
    "logit_selection_total_where_gold_in_mined",
]


def _row_with_schema(cols: List[str], values: Dict[str, Any]) -> Dict[str, Any]:
    return {c: values.get(c, None) for c in cols}


def cfg_to_prefixed_dict(cfg: RunConfig) -> Dict[str, Any]:
    return {
        "cfg_top_k": cfg.top_k,
        "cfg_dataset": cfg.dataset,
        "cfg_logit_max_mined_candidates": cfg.logit_max_mined_candidates,
        "cfg_logit_bias_strength": cfg.logit_bias_strength,
        "cfg_logit_max_token_logit_bias": cfg.logit_max_token_logit_bias,
        "cfg_logit_phrase_softmax_temperature": cfg.logit_phrase_softmax_temperature,
        "cfg_logit_clamp_first_line": int(cfg.logit_clamp_first_line) if cfg.logit_clamp_first_line is not None else None,
        "cfg_logit_hybrid_prompt": int(cfg.logit_hybrid_prompt) if cfg.logit_hybrid_prompt is not None else None,
        "cfg_logit_max_bias_steps": cfg.logit_max_bias_steps,
        "cfg_logit_bias_top_n": cfg.logit_bias_top_n,
    }


# =============================================================================
# Output parsing + diagnostics
# =============================================================================
@dataclass
class LogitDiag:
    """
    num_biased_token_ids:
        Number of unique token IDs that received logit bias.
        Indicates how broad vs focused the applied bias was.
    mined_hit:
        True if any mined phrase appears as a substring in the model prediction
        (after normalization). Weak signal that generation used mined content.
    gold_in_mined:
        True if any gold answer exactly matches a mined phrase (after normalization).
        Measures recall of the mining stage.
    oracle_em_from_mined:
        True if gold_in_mined. Represents the best-case exact-match outcome assuming
        an oracle could select from the mined candidates.
    selected_gold_given_gold_in_mined:
        True if the model prediction exactly matches a gold answer (after normalization)
        when the gold answer was present in mined candidates.
    """

    num_biased_token_ids: int = 0 
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
    mined_candidates: List[Tuple[str, float]]
    num_biased_token_ids: int
    top_biased_tokens: List[Tuple[int, float]]

def parse_model_out(out: Dict[str, Any]) -> ModelOut:
    extra = out.get("extra") or {}
    timings = out.get("timings_s") or {}
    pred = (out.get("raw_answer") or out.get("answer") or "").strip()

    # read from out first, then extra
    cache_used = bool(out.get("cache_used", extra.get("cache_used", False)))
    cache_hits = int(out.get("cache_hits", extra.get("cache_hits", 0)) or 0)
    cache_misses = int(out.get("cache_misses", extra.get("cache_misses", 0)) or 0)

    mined_raw = extra.get("mined_candidates") or out.get("mined_candidates") or extra.get("mined_phrases") or out.get("mined_phrases")
    mined_candidates = _coerce_mined_candidates(mined_raw)

    top_biased_tokens = extra.get("top_biased_tokens") or out.get("top_biased_tokens") or []

    num_biased_token_ids = int(
        extra.get("num_biased_token_ids")
        or out.get("num_biased_token_ids")
        or extra.get("bias_tokens")
        or out.get("bias_tokens")
        or 0
    )

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
        mined_candidates=mined_candidates,
        num_biased_token_ids=num_biased_token_ids,
        top_biased_tokens=top_biased_tokens,
    )


def compute_logit_diag(mo: ModelOut, golds: List[str], mode: str) -> LogitDiag:
    if mode != "logit_rag":
        return LogitDiag()

    phrases_only = [phrase for phrase, _ in mo.mined_candidates]
    mined_norm = [normalize_answer(x) for x in phrases_only]
    mined_norm = [x for x in mined_norm if x]

    pred_norm = normalize_answer(mo.pred)
    mined_hit = any(m in pred_norm for m in mined_norm if m)

    gold_norms = [normalize_answer(str(g)) for g in golds]
    gold_norms = [g for g in gold_norms if g]

    mined_set = set(mined_norm)
    gold_in_mined = any(g in mined_set for g in gold_norms)

    oracle = gold_in_mined
    selected_given_oracle = oracle and any(pred_norm == g for g in gold_norms)

    return LogitDiag(
        num_biased_token_ids=int(mo.num_biased_token_ids),
        mined_hit=mined_hit,
        gold_in_mined=gold_in_mined,
        oracle_em_from_mined=oracle,
        selected_gold_given_gold_in_mined=selected_given_oracle,
    )


# =============================================================================
# Runner (single dispatch)
# =============================================================================
def run_query(pipeline: RagPipeline, q: str, cfg: RunConfig) -> Dict[str, Any]:
    mode = validate_mode(cfg.mode)

    if mode == "logit_rag":
        return pipeline.run_logit_rag(
            q,
            max_mined_candidates=int(cfg.logit_max_mined_candidates),
            logit_bias_strength=float(cfg.logit_bias_strength),
            max_token_logit_bias=float(cfg.logit_max_token_logit_bias),
            phrase_softmax_temperature=float(cfg.logit_phrase_softmax_temperature),
            clamp_first_line=bool(cfg.logit_clamp_first_line),
            hybrid_prompt=bool(cfg.logit_hybrid_prompt),
            max_bias_steps=cfg.logit_max_bias_steps,
            bias_top_n=cfg.logit_bias_top_n,
        )

    # For llm / prompt_rag, your pipeline uses run_prompt_rag; llm should use top_k=0.
    return pipeline.run_prompt_rag(q)


# =============================================================================
# Per-example result
# =============================================================================
@dataclass
class ExampleResult:
    ex_id: str
    question: str
    golds: List[str]
    prediction: str
    em: float
    f1: float

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    embed_s: float
    ann_s: float
    docstore_s: float
    prompt_build_s: float
    decode_s: float
    total_s: float
    candidate_mine_s: float

    retrieved_doc_ids: List[Any]

    cache_used: bool
    cache_hits: int
    cache_misses: int

    logit: LogitDiag
    mined_candidates: List[Tuple[str, float]]  # full list (for debug JSONL / top10 preview)
    top_biased_tokens: List[Tuple[int, float]]  # top biased tokens (for debugging)


def evaluate_example(*, pipeline: RagPipeline, ex: Dict[str, Any], cfg: RunConfig) -> Optional[ExampleResult]:
    ex_id = ex.get("id", "")
    q, golds = get_question_and_answers(ex)
    if not q:
        return None

    out = run_query(pipeline, q, cfg)
    mo = parse_model_out(out)

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

    mode = validate_mode(cfg.mode)
    logit_diag = compute_logit_diag(mo, golds, mode)

    return ExampleResult(
        ex_id=str(ex_id),
        question=q,
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
        retrieved_doc_ids=mo.retrieved_doc_ids,
        cache_used=bool(mo.cache_used),
        cache_hits=int(mo.cache_hits),
        cache_misses=int(mo.cache_misses),
        logit=logit_diag,
        mined_candidates=mo.mined_candidates,
        top_biased_tokens=mo.top_biased_tokens,
    )


# =============================================================================
# Aggregator
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

    # logit
    logit_n: int = 0
    mined_hit_sum: int = 0
    gold_in_mined_sum: int = 0
    oracle_hits: int = 0
    oracle_total: int = 0
    select_hits_given_oracle: int = 0
    select_total_given_oracle: int = 0

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

        if mode == "logit_rag":
            self.logit_n += 1
            self.mined_hit_sum += int(res.logit.mined_hit)
            self.gold_in_mined_sum += int(res.logit.gold_in_mined)
            self.oracle_total += 1
            self.oracle_hits += int(res.logit.oracle_em_from_mined)

            if res.logit.oracle_em_from_mined:
                self.select_total_given_oracle += 1
                self.select_hits_given_oracle += int(res.logit.selected_gold_given_gold_in_mined)

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
        if mode == "logit_rag" and self.logit_n:
            post["mined_hit"] = f"{(self.mined_hit_sum/self.logit_n):.2f}"
            post["gold_in_m"] = f"{(self.gold_in_mined_sum/self.logit_n):.2f}"
            post["oracle"] = f"{(self.oracle_hits/self.oracle_total):.2f}" if self.oracle_total else "0.00"
            post["sel|oracle"] = (
                f"{(self.select_hits_given_oracle/self.select_total_given_oracle):.2f}"
                if self.select_total_given_oracle else "0.00"
            )
        return post


# =============================================================================
# Row building
# =============================================================================
def _topk_candidates(cands: List[Tuple[str, float]], k: int = 10) -> List[Tuple[str, float]]:
    # sort high->low by score; stable for ties
    return sorted(cands, key=lambda x: float(x[1]), reverse=True)[:k]


def build_result_row(*, run_id: str, cfg: RunConfig, res: ExampleResult) -> Dict[str, Any]:
    cfg_vals = cfg_to_prefixed_dict(cfg)
    top10 = _topk_candidates(res.mined_candidates, 10)

    values: Dict[str, Any] = {
        "run_id": run_id,
        "mode": cfg.mode,
        **cfg_vals,
        "id": res.ex_id,
        "question": res.question,
        "prediction": res.prediction,
        "golden_answers": json.dumps(res.golds, ensure_ascii=False),
        "em": float(res.em),
        "f1": float(res.f1),
        "retrieved_count": len(res.retrieved_doc_ids),
        "retrieved_doc_ids": json.dumps(res.retrieved_doc_ids),
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "total_tokens": res.total_tokens,
        "embed_s": res.embed_s,
        "ann_s": res.ann_s,
        "docstore_s": res.docstore_s,
        "prompt_build_s": res.prompt_build_s,
        "decode_s": res.decode_s,
        "total_s": res.total_s,
        "candidate_mine_s": res.candidate_mine_s,
        "cache_used": int(bool(res.cache_used)),
        "cache_hits": int(res.cache_hits),
        "cache_misses": int(res.cache_misses),
        "num_biased_token_ids": int(res.logit.num_biased_token_ids),
        "mined_hit": int(res.logit.mined_hit),
        "gold_in_mined": int(res.logit.gold_in_mined),
        "oracle_em_from_mined": int(res.logit.oracle_em_from_mined),
        "selected_gold_given_gold_in_mined": int(res.logit.selected_gold_given_gold_in_mined),
        "mined_candidates_count": len(res.mined_candidates),
        "mined_candidates_top10_json": json.dumps(top10, ensure_ascii=False),
        "top_biased_tokens_json": json.dumps(res.top_biased_tokens, ensure_ascii=False),
    }

    return _row_with_schema(RESULT_COLUMNS, values)


# =============================================================================
# Core evaluation
# =============================================================================
def evaluate_one_run(
    pipeline: RagPipeline,
    examples: List[Dict[str, Any]],
    *,
    cfg: RunConfig,
    results_csv: Optional[str],
    save_to_file: bool = True,
    debug_jsonl: Optional[str] = None,
    print_topk_candidates: int = 0,  # e.g. 10 prints top 10 mined candidates for first N samples
    print_top_tokens: int = 0,  # e.g. 5 prints top 5 biased tokens for first N samples
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

        # console preview for first N examples
        if agg.n <= cfg.print_first_n:
            tqdm.write("\n---")
            tqdm.write(f"run_id: {run_id}  mode: {mode}  id: {res.ex_id}")
            tqdm.write(f"Q: {res.question}")
            tqdm.write(f"PRED: {res.prediction}")
            tqdm.write(f"GOLDS: {res.golds}")
            tqdm.write(f"EM={res.em} F1={res.f1:.3f}")
            tqdm.write(f"tokens: prompt={res.prompt_tokens} completion={res.completion_tokens} total={res.total_tokens}")
            tqdm.write(f"timings_s: total={res.total_s:.3f} decode={res.decode_s:.3f}")

            if print_topk_candidates and mode == "logit_rag":
                topk = _topk_candidates(res.mined_candidates, print_topk_candidates)
                preview = ", ".join([f"{p}({s:.2f})" for p, s in topk])
                tqdm.write(f"mined_top{print_topk_candidates}: {preview}")

            if print_top_tokens and mode == "logit_rag":
                # This assumes that the logit bias was applied to the top-N mined candidates; adjust if your pipeline differs.
                topk = res.top_biased_tokens[:print_top_tokens]
                tqdm.write(f"top {print_top_tokens} biased tokens: {topk}")
                # If you have the specific token IDs and their bias values, you could print them here as well.


        if cfg.tqdm_update_every > 0 and agg.n % cfg.tqdm_update_every == 0:
            pbar.set_postfix(agg.postfix(mode))

        if save_to_file and results_csv:
            row = build_result_row(run_id=run_id, cfg=cfg, res=res)
            append_csv_row(results_csv, row)

        # optional full debug JSONL (per-example)
        if debug_jsonl:
            append_jsonl(
                debug_jsonl,
                {
                    "run_id": run_id,
                    "mode": mode,
                    "id": res.ex_id,
                    "question": res.question,
                    "prediction": res.prediction,
                    "golds": res.golds,
                    "em": res.em,
                    "f1": res.f1,
                    "retrieved_doc_ids": res.retrieved_doc_ids,
                    "mined_candidates": res.mined_candidates,  # full list
                    "top_biased_tokens": res.top_biased_tokens,  # full list
                },
            )

    cfg_vals = cfg_to_prefixed_dict(cfg)

    summary_values: Dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        **cfg_vals,
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
        "logit_mined_hit_rate": (agg.mined_hit_sum / agg.logit_n) if agg.logit_n else 0.0,
        "logit_gold_in_mined_rate": (agg.gold_in_mined_sum / agg.logit_n) if agg.logit_n else 0.0,
        "logit_oracle_em_from_mined_rate": (agg.oracle_hits / agg.oracle_total) if agg.oracle_total else 0.0,
        "logit_selection_accuracy_given_gold_in_mined": (
            (agg.select_hits_given_oracle / agg.select_total_given_oracle)
            if agg.select_total_given_oracle else 0.0
        ),
        "logit_selection_total_where_gold_in_mined": agg.select_total_given_oracle,
    }

    return _row_with_schema(SUMMARY_COLUMNS, summary_values)


# =============================================================================
# CLI + Hydra entry
# =============================================================================
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    parser = argparse.ArgumentParser()

    # modes: logit_rag, prompt_rag, llm

    #data/datasets/qa/nq/nq_train.jsonl
    #data/datasets/nq_train_filtered.jsonl
    parser.add_argument("--mode", type=str, default="llm", choices=sorted(VALID_MODES))
    parser.add_argument("--data", default="data/datasets/qa/nq/nq_train.jsonl", type=str)

    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--print_first_n", type=int, default=15)
    parser.add_argument("--tqdm_update_every", type=int, default=10)

    parser.add_argument("--results_csv", type=str, default="all_results.csv")
    parser.add_argument("--summary_csv", type=str, default="run_summaries.csv")
    parser.add_argument("--save_to_file", action="store_true", default=True)

    # optional debug output
    parser.add_argument("--debug_jsonl", type=str, default="", help="If set, append per-example debug records here.")
    parser.add_argument("--print_top_candidates", type=int, default=5, help="If >0, print top-K mined candidates for the first N examples.")
    parser.add_argument("--print_top_tokens", type=int, default=5, help="If >0, print top-K biased tokens for the first N examples.")

    # logit-only knobs
    parser.add_argument("--max_mined_candidates", type=int, default=40)
    parser.add_argument("--bias_top_n", type=int, default=5)

    # Single-flag sweeps: pass either "0.8" or "0.2,0.5,1.0"
    parser.add_argument("--logit_bias_strength", type=str, default="10")
    parser.add_argument("--max_token_logit_bias", type=str, default="0", help="If 0, no cap on token logit bias; otherwise, cap at this value")
    parser.add_argument("--phrase_softmax_temperature", type=str, default="20", help="Temperature for softmax over mined candidate scores when computing logit bias weights. Higher values make the distribution more uniform, lower values make it peakier.")

    parser.add_argument("--clamp_first_line", action="store_true")
    parser.add_argument("--hybrid_prompt", action="store_true")
    parser.add_argument("--max_bias_steps", type=int, default=None)

    parser.add_argument("--sweep_style", type=str, default="grid", choices=["grid", "zip"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("evaluate_dataset")

    mode = validate_mode(args.mode)

    # Configure pipeline
    if mode == "llm":
        top_k = 0
        cfg.prompt_build_method = "LLM_ONLY"
    elif mode == "prompt_rag":
        top_k = 10
        cfg.prompt_build_method = "QA_OPEN"
    else:
        top_k = 10
        cfg.prompt_build_method = "LOGIT_RAG"  # ensure your pipeline supports this method name

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

    # Sweep parsing (single flag -> list)
    strength_list = parse_float_list(args.logit_bias_strength)
    cap_list = parse_float_list(args.max_token_logit_bias)
    temp_list = parse_float_list(args.phrase_softmax_temperature)

    # combos: zip or grid
    if mode == "logit_rag":
        if args.sweep_style == "zip":
            k = max(len(strength_list), len(cap_list), len(temp_list))

            def pad(xs: List[float], k_: int) -> List[float]:
                return xs + [xs[-1]] * (k_ - len(xs))

            strength_list = pad(strength_list, k)
            cap_list = pad(cap_list, k)
            temp_list = pad(temp_list, k)
            combos = list(zip(strength_list, cap_list, temp_list))
        else:
            combos = [(s, c, t) for s in strength_list for c in cap_list for t in temp_list]
    else:
        # non-logit runs ignore these; still produce exactly one run
        combos = [(strength_list[0], cap_list[0], temp_list[0])]

    runs: List[RunConfig] = []
    for strength, cap, temp in combos:
        runs.append(
            RunConfig(
                mode=mode,
                dataset=args.data,
                top_k=top_k,
                limit=(args.limit if args.limit and args.limit > 0 else None),
                print_first_n=args.print_first_n,
                tqdm_update_every=args.tqdm_update_every,
                logit_max_mined_candidates=args.max_mined_candidates if mode == "logit_rag" else None,
                logit_bias_strength=float(strength) if mode == "logit_rag" else None,
                logit_max_token_logit_bias=float(cap) if mode == "logit_rag" else None,
                logit_phrase_softmax_temperature=float(temp) if mode == "logit_rag" else None,
                logit_clamp_first_line=bool(args.clamp_first_line) if mode == "logit_rag" else None,
                logit_hybrid_prompt=bool(args.hybrid_prompt) if mode == "logit_rag" else None,
                logit_max_bias_steps=args.max_bias_steps if mode == "logit_rag" else None,
                logit_bias_top_n=args.bias_top_n if mode == "logit_rag" else None,
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

    debug_jsonl = args.debug_jsonl.strip() or None
    if debug_jsonl:
        logger.info("Debug JSONL enabled: %s", debug_jsonl)

    for rc in runs:
        logger.info("RUN %s cfg=%s", rc.run_id(), json.dumps(asdict(rc), default=str))

        summary = evaluate_one_run(
            pipeline,
            examples,
            cfg=rc,
            results_csv=args.results_csv,
            save_to_file=args.save_to_file,
            debug_jsonl=debug_jsonl,
            print_topk_candidates=int(args.print_top_candidates or 0),
            print_top_tokens=int(args.print_top_tokens or 0),
        )

        print("\n==== FINAL (run_id={}) ====".format(rc.run_id()))
        print(json.dumps(summary, indent=2, default=str))

        if args.summary_csv:
            append_csv_row(args.summary_csv, summary)
            logger.info("Appended summary row to %s (run_id=%s)", args.summary_csv, summary.get("run_id"))


if __name__ == "__main__":
    main()
