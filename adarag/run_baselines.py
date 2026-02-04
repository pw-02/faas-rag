"""
Run baseline experiments for open-domain QA:

1) No retrieval (k=0)
2) Fixed-k RAG for k in {1, 5, 20} (configurable)

Reports:
- EM, token-F1
- Avg generated tokens
- Avg retrieval calls (0 for no-retrieval, 1 for fixed-k RAG)
Optionally prints sample examples and saves outputs.

Usage:
python run_baselines.py \
  --faiss_index /path/to/wiki.index \
  --passage_store /path/to/passages.json \
  --encoder sentence-transformers/multi-qa-mpnet-base-dot-v1 \
  --generator meta-llama/Meta-Llama-3-8B-Instruct \
  --data /path/to/dev.jsonl \
  --ks 0 1 5 20 \
  --limit 500 \
  --max_ctx_chars 4000 \
  --max_new_tokens 64 \
  --device cuda \
  --show_examples 5 \
  --save_dir /tmp/baseline_outputs
"""

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import boto3

# -------------------------
# Metrics
# -------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


# -------------------------
# Data
# -------------------------

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            ex = json.loads(line)
            if "question" not in ex or "answer" not in ex:
                raise ValueError("Each jsonl line must contain fields: question, answer")
            data.append({"question": ex["question"], "answer": ex["answer"]})
    return data


# -------------------------
# Retriever
# -------------------------

@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float

class STRetriever:
    """
    SentenceTransformer dense retriever over FAISS + JSON store.
    Assumes:
      - passage_store is dict keyed by str(pid)
      - embeddings NOT normalized (dot-product retrieval)
    """
    def __init__(self, faiss_index_path: str, passage_store_path: str,
                 encoder_name_or_path: str, device: str = "cuda"):
        self.index = faiss.read_index(faiss_index_path)
        with open(passage_store_path, "r", encoding="utf-8") as f:
            self.passages: Dict[str, Any] = json.load(f)

        self.qenc = SentenceTransformer(encoder_name_or_path, device=device)

        # dim check
        v = self.encode_question("dimension check")
        if v.shape[0] != self.index.d:
            raise ValueError(
                f"Encoder dim {v.shape[0]} != FAISS dim {self.index.d}. "
                "You likely used a different encoder to build the index."
            )

        print(f"[Retriever] index dim={self.index.d} metric_type={getattr(self.index, 'metric_type', 'unknown')}")
        print("[Retriever] Using normalize_embeddings=False (dot product).")

    def encode_question(self, question: str) -> np.ndarray:
        vec = self.qenc.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=False,   # IMPORTANT per your setup
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def retrieve(self, question: str, k: int) -> List[Passage]:
        q = self.encode_question(question).reshape(1, -1)
        scores, idxs = self.index.search(q, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Passage] = []
        for score, pid in zip(scores, idxs):
            meta = self.passages.get(str(pid))
            if meta is None:
                continue
            title = meta.get("title", "") or ""
            text = meta.get("text", "") or meta.get("contents", "") or meta.get("passage", "") or ""
            out.append(Passage(pid=int(pid), title=title, text=text, score=float(score)))
        return out


# -------------------------
# Prompting
# -------------------------

def format_context(passages: List[Passage], max_chars: int) -> str:
    ctx = ""
    for p in passages:
        block = f"[{p.pid}] {p.title}\n{p.text}\n\n"
        if len(ctx) + len(block) > max_chars:
            break
        ctx += block
    return ctx.strip()

def prompt_no_retrieval(question: str) -> str:
    return (
        "Answer the question. If you are not sure, say \"I don't know\".\n\n"
        f"Question: {question}\nAnswer:"
    )

def prompt_with_context(question: str, passages: List[Passage], max_ctx_chars: int) -> str:
    ctx = format_context(passages, max_ctx_chars)
    return (
        "Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\nAnswer:"
    )


# -------------------------
# Generator
# -------------------------

class LocalGenerator:
    def __init__(self, model_name_or_path: str, device: str = "cuda", max_new_tokens: int = 64):
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device.startswith("cuda") else None
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[str, int]:
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id
        )
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return text, int(gen_ids.shape[-1])


# -------------------------
# Baseline runner
# -------------------------

def run_setting(
    examples: List[Dict[str, str]],
    retriever: STRetriever,
    gen: LocalGenerator,
    k: int,
    max_ctx_chars: int,
    show_examples: int,
    seed: int = 0,
) -> Dict[str, Any]:
    ems, f1s = [], []
    total_gen_tokens = 0
    total_ret_calls = 0
    outputs = []

    # pick which indices to print
    random.seed(seed)
    show_idxs = set(random.sample(range(len(examples)), k=min(show_examples, len(examples))))

    for i, ex in enumerate(tqdm(examples, desc=f"Run k={k}")):
        q, gold = ex["question"], ex["answer"]

        if k == 0:
            prompt = prompt_no_retrieval(q)
            passages = []
            ret_calls = 0
        else:
            passages = retriever.retrieve(q, k=k)
            prompt = prompt_with_context(q, passages, max_ctx_chars=max_ctx_chars)
            ret_calls = 1

        pred, gen_tokens = gen.generate(prompt)

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        ems.append(em)
        f1s.append(f1)
        total_gen_tokens += gen_tokens
        total_ret_calls += ret_calls

        record = {
            "i": i,
            "k": k,
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "gen_tokens": gen_tokens,
            "retrieval_calls": ret_calls,
        }
        if k != 0:
            record["top_passages"] = [
                {"pid": p.pid, "score": p.score, "title": p.title}
                for p in passages[:3]
            ]
        outputs.append(record)

        if i in show_idxs:
            print("\n" + "=" * 90)
            print(f"k={k}")
            print("Q:", q)
            print("Gold:", gold)
            print("Pred:", pred)
            print(f"EM={em:.0f}  F1={f1:.3f}  gen_tokens={gen_tokens}  ret_calls={ret_calls}")
            if k != 0:
                print("\nTop retrieved passages:")
                for p in passages[:3]:
                    snippet = (p.text[:220] + "...") if len(p.text) > 220 else p.text
                    print(f"  pid={p.pid} score={p.score:.4f} title={p.title}")
                    print("   ", snippet.replace("\n", " "))

    N = max(1, len(examples))
    return {
        "k": k,
        "N": len(examples),
        "EM": float(np.mean(ems)) if ems else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_gen_tokens": float(total_gen_tokens / N),
        "avg_retrieval_calls": float(total_ret_calls / N),
        "outputs": outputs,
    }

def download_index_from_s3(s3_bucket: str, s3_prefix: str, index:str, local_dir: str, skip_existing: bool = True):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix + '/' + index)
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            rel_path = os.path.relpath(key, s3_prefix)
            local_path = os.path.join(local_dir, rel_path)
            if skip_existing and os.path.exists(local_path):
                print(f"Skipping existing file: {local_path}")
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading s3://{s3_bucket}/{key} to {local_path}")
            s3.download_file(s3_bucket, key, local_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3_bucket", required=False, default='vectorindexes')
    ap.add_argument("--s3_prefix", required=False, default='wiki-dpr/faiss_wiki_dpr')
    ap.add_argument("--local_dir", required=False, default='data/indexes/wiki-dpr')
    ap.add_argument("--faiss_index", required=True, choices=["flat_100k", "hnsw_100k"])
    
    ap.add_argument("--passage_store", required=False, default='data/indexes/wiki-dpr/passages.json')
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 1, 5, 20])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--show_examples", type=int, default=5)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    download_index_from_s3(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        index=args.faiss_index,
        local_dir=args.local_dir,
        skip_existing=True
    )



    examples = load_jsonl(args.data, limit=args.limit)

    retriever = STRetriever(
        faiss_index_path=args.faiss_index,
        passage_store_path=args.passage_store,
        encoder_name_or_path=args.encoder,
        device=args.device,
    )
    gen = LocalGenerator(
        model_name_or_path=args.generator,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    all_results = []
    for k in args.ks:
        res = run_setting(
            examples=examples,
            retriever=retriever,
            gen=gen,
            k=k,
            max_ctx_chars=args.max_ctx_chars,
            show_examples=args.show_examples,
            seed=args.seed,
        )
        all_results.append(res)

    # Print summary table
    print("\n" + "#" * 90)
    print("Summary (higher EM/F1 better, lower cost better)")
    print(f"{'k':>4} {'N':>6} {'EM':>8} {'F1':>8} {'avg_gen_tok':>12} {'avg_ret_calls':>14}")
    for r in all_results:
        print(f"{r['k']:>4} {r['N']:>6} {r['EM']:>8.4f} {r['F1']:>8.4f} {r['avg_gen_tokens']:>12.1f} {r['avg_retrieval_calls']:>14.3f}")

    # Save
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        summary_path = os.path.join(args.save_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full results to: {summary_path}")

        # also save a compact CSV-like json for quick reading
        compact = [{
            "k": r["k"],
            "N": r["N"],
            "EM": r["EM"],
            "F1": r["F1"],
            "avg_gen_tokens": r["avg_gen_tokens"],
            "avg_retrieval_calls": r["avg_retrieval_calls"],
        } for r in all_results]
        compact_path = os.path.join(args.save_dir, "compact.json")
        with open(compact_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)
        print(f"Saved compact summary to: {compact_path}")


if __name__ == "__main__":
    main()
