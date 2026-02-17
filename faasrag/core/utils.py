import collections
import csv
import os
import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Tuple
















def append_csv_row(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


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
