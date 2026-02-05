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
  --s3_bucket vectorindexes \
  --s3_prefix wiki-dpr \
  --faiss_index faiss_wiki_dpr/flat_100k \
  --passage_store passages/100k \
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
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import boto3
import datetime

# -------------------------
# Metrics
# -------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, golds: Union[str, List[str]]) -> float:
    """
    pred: model prediction string
    golds: either a string or list of acceptable gold answers
    """
    if isinstance(golds, str):
        golds = [golds]

    pred_n = normalize_text(pred)
    for g in golds:
        if pred_n == normalize_text(g):
            return 1.0
    return 0.0

def token_f1(pred: str, golds: Union[str, List[str]]) -> float:
    """
    Returns max token-F1 over gold answers.
    """
    if isinstance(golds, str):
        golds = [golds]

    pred_tokens = normalize_text(pred).split()
    if len(pred_tokens) == 0:
        return 0.0

    from collections import Counter
    pred_cnt = Counter(pred_tokens)

    best_f1 = 0.0

    for g in golds:
        gold_tokens = normalize_text(g).split()
        if len(gold_tokens) == 0:
            continue

        gold_cnt = Counter(gold_tokens)
        common = sum((pred_cnt & gold_cnt).values())

        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1

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
            if "question" not in ex or "golden_answers" not in ex:
                raise ValueError("Each jsonl line must contain fields: question, answer")
            data.append({"question": ex["question"], "answer": ex["golden_answers"]})
    return data

# -------------------------
# S3 download helpers
# -------------------------

def s3_download_prefix(s3_bucket: str, prefix: str, local_dir: str, skip_existing: bool = True):
    """
    Download all objects under s3://bucket/prefix into local_dir/<key> (mirrors S3 keys).
    Example:
      bucket=vectorindexes, prefix="wiki-dpr/faiss_wiki_dpr/flat_100k"
      -> downloads to local_dir/wiki-dpr/faiss_wiki_dpr/flat_100k/...
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    found_any = False
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found_any = True
            key = obj["Key"]
            local_path = os.path.join(local_dir, key)  # mirror full key under local_dir

            if skip_existing and os.path.exists(local_path):
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading s3://{s3_bucket}/{key} -> {local_path}")
            s3.download_file(s3_bucket, key, local_path)

    if not found_any:
        raise FileNotFoundError(f"No objects found at s3://{s3_bucket}/{prefix}")

def find_first_file(root: str, exts: Tuple[str, ...]) -> str:
    matches = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                matches.append(os.path.join(r, fn))
    if not matches:
        raise FileNotFoundError(f"No files ending with {exts} under {root}")
    return sorted(matches)[0]

def resolve_faiss_index_path(local_index_dir_or_file: str) -> str:
    """
    faiss.read_index needs a file. If a directory is provided, pick a likely index file.
    """
    if os.path.isfile(local_index_dir_or_file):
        return local_index_dir_or_file

    if not os.path.isdir(local_index_dir_or_file):
        raise FileNotFoundError(f"FAISS index path not found: {local_index_dir_or_file}")

    # Prefer common index extensions
    try:
        return find_first_file(local_index_dir_or_file, (".index", ".faiss"))
    except FileNotFoundError:
        # Sometimes the file might be named literally "index"
        index_path = os.path.join(local_index_dir_or_file, "index")
        if os.path.isfile(index_path):
            return index_path
        raise

def resolve_passage_store_path(local_passages_dir_or_file: str) -> str:
    """
    Your retriever expects a JSON file (dict keyed by pid). If given a directory, pick first .json.
    """
    if os.path.isfile(local_passages_dir_or_file):
        return local_passages_dir_or_file

    if not os.path.isdir(local_passages_dir_or_file):
        raise FileNotFoundError(f"Passage store path not found: {local_passages_dir_or_file}")

    return find_first_file(local_passages_dir_or_file, (".jsonl",))

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
    SentenceTransformer dense retriever over FAISS + JSONL store.

    Assumes:
      - passage_store is JSONL (one passage per line)
      - each line contains pid (or id) plus title/text fields
      - embeddings NOT normalized (dot-product retrieval)
    """
    def __init__(
        self,
        faiss_index_path: str,
        passage_store_path: str,
        encoder_name_or_path: str,
        device: str = "cuda",
    ):
        self.index = faiss.read_index(faiss_index_path)
        # After loading the index
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = 128  # try 128 or 256 for better recall

        # ---- load JSONL passage store into dict keyed by str(pid) ----
        self.passages: Dict[str, Any] = {}
        with open(passage_store_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} in {passage_store_path}: {e}"
                    ) from e

                pid = obj.get("faiss_id", None)
                if pid is None:
                    pid = obj.get("faiss_id", None)
                if pid is None:
                    # If your JSONL uses a different key, add it here
                    raise ValueError(
                        f"Passage JSON missing faiss_id on line {line_no} in {passage_store_path}. "
                        f"Keys present: {list(obj.keys())[:20]}"
                    )

                self.passages[str(pid)] = obj

        print(f"[Retriever] Loaded {len(self.passages)} passages from JSONL")

        # ---- encoder ----
        self.qenc = SentenceTransformer(encoder_name_or_path, device=device)

        # ---- dim check ----
        v = self.encode_question("dimension check")
        if v.shape[0] != self.index.d:
            raise ValueError(
                f"Encoder dim {v.shape[0]} != FAISS dim {self.index.d}. "
                "You likely used a different encoder to build the index."
            )

        print(f"[Retriever] index file={faiss_index_path}")
        print(f"[Retriever] store file={passage_store_path}")
        print(f"[Retriever] index dim={self.index.d} metric_type={getattr(self.index, 'metric_type', 'unknown')}")
        print("[Retriever] Using normalize_embeddings=False (dot product).")

    def encode_question(self, question: str) -> np.ndarray:
        vec = self.qenc.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=False,  # IMPORTANT per your setup
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
    return [
        {"role": "system", "content": "You answer questions with a short answer only."},
        {"role": "user", "content": f'Answer with ONLY the answer. If unsure, say "I don\'t know".\n\nQuestion: {question}'},
    ]

def prompt_with_context(question: str, passages: List[Passage], max_ctx_chars: int) -> str:
    context = format_context(passages, max_ctx_chars)
    return [
        {"role": "system", "content": "You answer questions using the provided context. Output a short answer only."},
        {"role": "user", "content": f'Use ONLY this context. Answer with ONLY the answer. If unsure, say "I don\'t know".\n\nContext:\n{context}\n\nQuestion: {question}'},
    ]

def extract_short_answer(text: str) -> str:
    t = text.strip()

    # take first line
    t = t.split("\n")[0].strip()

    # take first sentence
    t = t.split(".")[0].strip()

    # clean quotes
    t = t.strip('"').strip("'").strip()

    return t

# -------------------------
# Generator
# -------------------------

class LocalGenerator:
    def __init__(self, model_name_or_path: str, 
                 device: str = None,
                 max_new_tokens: int = 64):
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # device_map="auto" loads across GPUs if available; otherwise loads on one GPU.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device.startswith("cuda") else None,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, messages) -> Tuple[str, int]:
        # Build chat-formatted prompt
        prompt = self.tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        out = self.model.generate(
            **inputs,
            max_new_tokens=min(self.max_new_tokens, 16),  # short answers for QA eval
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.15,        # prevents looping
            no_repeat_ngram_size=3,          # extra safety
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        )

        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        raw_text = self.tok.decode(gen_ids, skip_special_tokens=True)

        text = extract_short_answer(raw_text)
        return text, int(gen_ids.shape[-1])


# -------------------------
# Baseline runner
# -------------------------

def run_setting(
    examples: List[Dict[str, str]],
    retriever: Optional[STRetriever],
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

    random.seed(seed)
    show_idxs = set(random.sample(range(len(examples)), k=min(show_examples, len(examples))))

    for i, ex in enumerate(tqdm(examples, desc=f"Run k={k}")):
        q, gold = ex["question"], ex["answer"]

        if k == 0:
            prompt = prompt_no_retrieval(q)
            passages = []
            ret_calls = 0
        else:
            if retriever is None:
                raise RuntimeError("Retriever is required for k>0 but was not initialized.")
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
            record["top_passages"] = [{"pid": p.pid, "score": p.score, "title": p.title} for p in passages[:3]]
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

def infer_passage_store_from_index(faiss_index: str) -> str:
    if faiss_index.endswith("_100k"):
        return "wiki-passages/100k"
    if faiss_index.endswith("_500k"):
        return "wiki-passages/500k"
    if faiss_index.endswith("_1m"):
        return "wiki-passages/1m"
    if faiss_index.endswith("_2_5m"):
        return "wiki-passages/2_5m"
    if faiss_index.endswith("_5m"):
        return "wiki-passages/5m"
    if faiss_index.endswith("_10m"):
        return "wiki-passages/10m"
    raise ValueError(f"Unknown index size in faiss_index: {faiss_index}")


def main():



    run_date_time_now  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser()

    ap.add_argument("--s3_bucket", required=False, default="vectorindexes")
    ap.add_argument("--s3_prefix", required=False, default="wiki-dpr")

    # IMPORTANT: treat these as "paths relative to s3_prefix" (logical names), not local paths
    ap.add_argument(
        "--faiss_index",
        required=False,
        choices=[
            "faiss_wiki_dpr/flat_100k",
            "faiss_wiki_dpr/hnsw_100k",
            "faiss_wiki_dpr/ivf_100k",
            "faiss_wiki_dpr/flat_500k",
            "faiss_wiki_dpr/hnsw_500k",
            "faiss_wiki_dpr/ivf_500k",
            "faiss_wiki_dpr/flat_1m",
            "faiss_wiki_dpr/hnsw_1m",
            "faiss_wiki_dpr/ivf_1m",
            # "faiss_wiki_dpr/flat_2_5m",
            "faiss_wiki_dpr/hnsw_2_5m",
        ],
        default="faiss_wiki_dpr/hnsw_2_5m",
    )

    # Make passage_store optional; default=None means "infer it"
    ap.add_argument(
        "--passage_store",
        required=False,
        default=None,
        help="If omitted, inferred from --faiss_index (e.g. *_500k -> wiki-passages/500k).",
    )
    # local_dir is the root under which we mirror S3 keys
    ap.add_argument("--local_dir", required=False, default="data/indexes")
    ap.add_argument("--queries", required=False, default="data/datasets/qa/nq/nq_dev.jsonl")

    ap.add_argument("--encoder", required=False, default="facebook-dpr-question_encoder-single-nq-base")
    ap.add_argument("--generator", required=False, default="meta-llama/Llama-3.1-8B-Instruct", help="e.g., Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.1-8B-Instruct")

    ap.add_argument("--ks", nargs="+", type=int, default=[0, 1, 5, 20])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max_ctx_chars", type=int, default=4000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g., cpu, cuda")
    ap.add_argument("--show_examples", type=int, default=5)
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.passage_store is None:
        args.passage_store = infer_passage_store_from_index(args.faiss_index)

    # Decide whether we even need retrieval
    need_retrieval = any(k > 0 for k in args.ks)

    local_faiss_index_path = None
    local_passage_store_path = None

    if need_retrieval:
        # Download from S3
        index_prefix = f"{args.s3_prefix}/{args.faiss_index}".rstrip("/")
        passages_prefix = f"{args.s3_prefix}/{args.passage_store}".rstrip("/")

        s3_download_prefix(args.s3_bucket, index_prefix, args.local_dir, skip_existing=True)
        s3_download_prefix(args.s3_bucket, passages_prefix, args.local_dir, skip_existing=True)

        # Resolve local filesystem paths
        local_index_dir = os.path.join(args.local_dir, args.s3_prefix, args.faiss_index)
        local_passages_dir_or_file = os.path.join(args.local_dir, args.s3_prefix, args.passage_store)

        local_faiss_index_path = resolve_faiss_index_path(local_index_dir)
        local_passage_store_path = resolve_passage_store_path(local_passages_dir_or_file)

    examples = load_jsonl(args.queries, limit=args.limit)

    # Generator always needed
    gen = LocalGenerator(
        model_name_or_path=args.generator,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    # Retriever only if needed
    retriever = None
    if need_retrieval:
        retriever = STRetriever(
            faiss_index_path=local_faiss_index_path,
            passage_store_path=local_passage_store_path,
            encoder_name_or_path=args.encoder,
            device=args.device,
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
        print(
            f"{r['k']:>4} {r['N']:>6} {r['EM']:>8.4f} {r['F1']:>8.4f} "
            f"{r['avg_gen_tokens']:>12.1f} {r['avg_retrieval_calls']:>14.3f}"
        )

    # Save
    if not args.save_dir:
        #make save dir named ofter index, generator, and query file
        args.save_dir = f"adarag/results/{os.path.basename(args.faiss_index).rsplit('.',1)[0]}_{os.path.basename(args.generator).rsplit('.',1)[0]}_{os.path.basename(args.queries).rsplit('.',1)[0]}"

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        file_name = f"baseline_results_{os.path.basename(args.queries).rsplit('.',1)[0]}_{run_date_time_now}.json"
        summary_path = os.path.join(args.save_dir, file_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved full results to: {summary_path}")

        compact = [
            {
                "k": r["k"],
                "N": r["N"],
                "EM": r["EM"],
                "F1": r["F1"],
                "avg_gen_tokens": r["avg_gen_tokens"],
                "avg_retrieval_calls": r["avg_retrieval_calls"],
            }
            for r in all_results
        ]
        compact_path = os.path.join(args.save_dir, f"compact_{os.path.basename(args.queries).rsplit('.',1)[0]}_{run_date_time_now}.json")
        with open(compact_path, "w", encoding="utf-8") as f:
            json.dump(compact, f, ensure_ascii=False, indent=2)
        print(f"Saved compact summary to: {compact_path}")

if __name__ == "__main__":
    main()
