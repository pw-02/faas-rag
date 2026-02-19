import collections
import csv
import os
import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Tuple


def dedupe_overlapping_phrases(
    scored_phrases: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """
    Keep longer/more specific phrases; drop phrases that are substrings of an already-kept phrase
    (after simple normalization). Assumes scored_phrases are roughly best->worst, but we enforce it.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    # sort by score desc, then by normalized length desc
    ranked = sorted(scored_phrases, key=lambda x: (float(x[1]), len(norm(x[0]))), reverse=True)

    kept: List[Tuple[str, float]] = []
    kept_norm: List[str] = []
    for phrase, score in ranked:
        p = (phrase or "").strip()
        pn = norm(p)
        if not pn:
            continue
        # if this phrase is contained in any already-kept phrase, skip
        if any(pn in kn for kn in kept_norm):
            continue
        kept.append((p, float(score)))
        kept_norm.append(pn)


    return kept
def append_csv_row(path: str, row: dict) -> None:
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        # ✅ write header once
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def extract_short_answer(text: str, max_chars: Optional[int] = None) -> str:
    t = (text or "").strip()
    t = t.split("\n")[0].strip()
    # optional heuristic: first sentence (can be brittle)
    t = t.split(".")[0].strip()
    t = t.strip('"').strip("'").strip()
    if max_chars is not None:
        t = t[:max_chars].strip()
    return t


def normalize_answer(s: str) -> str:
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Iterable[str]):
    gts = list(ground_truths) if ground_truths is not None else [""]
    if not gts:
        gts = [""]
    return max(metric_fn(prediction, gt) for gt in gts)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: Iterable[str]):
    ground_truths = list(ground_truths) if ground_truths is not None else [""]
    if len(ground_truths) == 0:
        ground_truths = [""]
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores)


def oracle_em_from_mined(mined: List[str], golds: List[str]) -> bool:
    """
    True if ANY mined candidate exactly matches ANY gold answer (after normalization).
    """
    if not mined or not golds:
        return False
    mined_norm = {normalize_answer(m) for m in mined if str(m).strip()}
    gold_norm = {normalize_answer(g) for g in golds if str(g).strip()}
    return any(g in mined_norm for g in gold_norm)


def selection_accuracy_given_gold_in_mined(pred: str, mined: List[str], golds: List[str]) -> Tuple[bool, bool]:
    """
    Returns:
      (gold_in_mined, pred_is_gold)
    Where:
      gold_in_mined: oracle_em_from_mined(...)
      pred_is_gold: prediction exactly matches a gold (after normalization)
    """
    gold_in = oracle_em_from_mined(mined, golds)
    pred_is_gold = False
    if pred and golds:
        pred_is_gold = any(normalize_answer(pred) == normalize_answer(g) for g in golds)
    return gold_in, pred_is_gold


def parse_float_list(s: str) -> List[Optional[float]]:
    # supports "0.1,0.2,0.4" or "[0.1, 0.2]" or "0.1 0.2"
    if not s:
        return []

    s = s.strip().lower()
    if s in {"none", "null"}:
        return [None]

    s = s.strip("[]()")
    parts = [p.strip() for p in s.replace(",", " ").split()]

    out: List[Optional[float]] = []
    for p in parts:
        if not p:
            continue
        out.append(float(p))

    return out

def _topk_bias(bias: dict[int, float], k: int = 50):
    return sorted(bias.items(), key=lambda kv: kv[1], reverse=True)[:k]

def pretty_print_top_biased_tokens(tokenizer, top_bias, k: int = 20) -> str:
    lines = []
    for tid, val in top_bias[:k]:
        s = tokenizer.decode([int(tid)])
        lines.append(f"{repr(s):>16}  id={int(tid):<7}  bias={float(val):.4f}")
    return "\n".join(lines)

def top_biased_tokens_dict(tokenizer, bias: dict[int, float], k: int = 20):
    top_bias = _topk_bias(bias, k)
    out = []
    for tid, val in top_bias[:k]:
        out.append({
            "token_id": int(tid),
            "token": tokenizer.decode([int(tid)]),
            "bias": float(val),
        })
    return out
def top_biased_tokens_pairs(tokenizer, bias: dict[int, float], k: int = 20) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for tid, val in _topk_bias(bias, k):
        token = tokenizer.decode([int(tid)])
        out.append((token, float(val)))
    return out
