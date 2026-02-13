from logging import Logger
import collections
import json
import logging
import re
import string
import csv
import os

# Third Party
# from rouge_score import rouge_scorer
# ============================================================
# Output Normalization
# ============================================================
def append_csv_row(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def extract_short_answer(text: str, max_chars: int | None = None) -> str:
    """
    Normalize model output for QA-style evaluation:
    - first line
    - first sentence
    - strip quotes
    """
    t = text.strip()

    # first line
    t = t.split("\n")[0].strip()

    # first sentence
    t = t.split(".")[0].strip()

    # clean quotes
    t = t.strip('"').strip("'").strip()

    if max_chars is not None:
        t = t[:max_chars].strip()

    return t


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_generation(s):
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# def compute_rl(pred, gold):
#     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     rougeL = scorer.score(gold, pred)["rougeL"].fmeasure
#     return rougeL
