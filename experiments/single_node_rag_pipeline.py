#!/usr/bin/env python3
import argparse
import os
import time
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict, Tuple
import numpy as np
import faiss
import boto3


# -----------------------
# Timing helpers
# -----------------------
def now() -> float:
    return time.perf_counter()

def pct(arr: List[float], p: int) -> float:
    return float(np.percentile(np.array(arr, dtype=np.float64), p))

def print_stats(name: str, ms: List[float]):
    if not ms:
        print(f"{name}: no samples")
        return
    print(
        f"{name:>12}: "
        f"p50={pct(ms,50):.2f}ms  p95={pct(ms,95):.2f}ms  p99={pct(ms,99):.2f}ms  mean={float(np.mean(ms)):.2f}ms"
    )


# -----------------------
# Index loading (disk or S3->disk)
# -----------------------
def download_from_s3(bucket: str, key: str, local_path: str, region: str, force: bool):
    s3 = boto3.client("s3", region_name=region)
    if not force and os.path.exists(local_path):
        print(f"Using cached file: {local_path}")
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)

def load_faiss_index(path: str, mmap: bool) -> faiss.Index:
    flags = faiss.IO_FLAG_MMAP if mmap else 0
    print(f"Loading FAISS index: {path} (mmap={mmap})")
    return faiss.read_index(path, flags)

def load_index(args) -> faiss.Index:
    if args.index_source == "s3":
        if not args.s3_bucket or not args.s3_index_key:
            raise ValueError("--s3-bucket and --s3-index-key required when --index-source s3")
        download_from_s3(
            bucket=args.s3_bucket,
            key=args.s3_index_key,
            local_path=args.index_path,
            region=args.s3_region,
            force=args.force_download,
        )
    return load_faiss_index(args.index_path, mmap=args.mmap)


# -----------------------
# Doc stores
# -----------------------
class DocStore(Protocol):
    def get_many(self, row_ids: List[int]) -> List[str]:
        ...

@dataclass
class MemoryJsonlByRowDocStore:
    """
    Loads docs_by_row_*.jsonl into memory.
    Expected line format: {"row_id": <int>, "text": <str>}
    """
    jsonl_path: str
    texts: List[Optional[str]] = None

    def __post_init__(self):
        print(f"Loading docs JSONL into memory: {self.jsonl_path}")
        # Build a list where index == row_id
        max_row = -1
        temp: Dict[int, str] = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rid = int(obj["row_id"])
                temp[rid] = obj["text"]
                if rid > max_row:
                    max_row = rid
        self.texts = [None] * (max_row + 1)
        for rid, text in temp.items():
            self.texts[rid] = text
        print(f"Loaded {len(temp)} docs into memory (max_row_id={max_row}).")

    def get_many(self, row_ids: List[int]) -> List[str]:
        out = []
        for rid in row_ids:
            if 0 <= rid < len(self.texts):
                t = self.texts[rid]
                if t is not None:
                    out.append(t)
        return out

@dataclass
class SqliteByRowDocStore:
    """
    SQLite schema: docs(row_id INTEGER PRIMARY KEY, text TEXT)
    """
    db_path: str

    def __post_init__(self):
        print(f"Opening SQLite doc DB: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def get_many(self, row_ids: List[int]) -> List[str]:
        if not row_ids:
            return []
        q = ",".join("?" * len(row_ids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT row_id, text FROM docs WHERE row_id IN ({q})", row_ids)
        rows = cur.fetchall()
        m = {int(rid): text for rid, text in rows}
        return [m[rid] for rid in row_ids if rid in m]


def build_doc_store(args) -> Optional[DocStore]:
    if args.doc_store == "none":
        return None
    if args.doc_store == "memory_jsonl":
        if not args.docs_jsonl_path:
            raise ValueError("--docs-jsonl-path required for --doc-store memory_jsonl")
        return MemoryJsonlByRowDocStore(args.docs_jsonl_path)
    if args.doc_store == "sqlite":
        if not args.docs_db_path:
            raise ValueError("--docs-db-path required for --doc-store sqlite")
        return SqliteByRowDocStore(args.docs_db_path)
    raise ValueError(f"Unknown doc_store: {args.doc_store}")


# -----------------------
# Embedding backends
# -----------------------
class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...

@dataclass
class RandomEmbedder:
    dim: int
    cosine: bool

    def embed(self, texts: List[str]) -> np.ndarray:
        x = np.random.randn(len(texts), self.dim).astype(np.float32)
        if self.cosine:
            faiss.normalize_L2(x)
        return x

@dataclass
class SentenceTransformersEmbedder:
    model_name: str
    device: str
    cosine: bool

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer  # requires sentence-transformers
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        x = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=self.cosine).astype(np.float32)
        return x

def build_embedder(args) -> Embedder:
    if args.embed_backend == "random":
        return RandomEmbedder(dim=args.dim, cosine=args.cosine)
    if args.embed_backend == "sentence_transformers":
        if not args.embed_model:
            raise ValueError("--embed-model required for --embed-backend sentence_transformers")
        return SentenceTransformersEmbedder(model_name=args.embed_model, device=args.device, cosine=args.cosine)
    raise ValueError(f"Unknown embed_backend: {args.embed_backend}")


# -----------------------
# LLM inference backends
# -----------------------
class LLM(Protocol):
    def generate(self, prompts: List[str]) -> List[str]:
        ...

@dataclass
class DummyLLM:
    sleep_ms: float
    def generate(self, prompts: List[str]) -> List[str]:
        time.sleep(self.sleep_ms / 1000.0)
        return ["<dummy>" for _ in prompts]

@dataclass
class TransformersLLM:
    model_name: str
    device: str
    max_new_tokens: int
    temperature: float

    def __post_init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=(torch.float16 if self.device.startswith("cuda") else None),
            device_map=("auto" if self.device.startswith("cuda") else None),
        )
        if not self.device.startswith("cuda"):
            self.model.to(self.device)

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=(self.temperature > 0),
                    temperature=self.temperature if self.temperature > 0 else None,
                )
            txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
            outputs.append(txt)
        return outputs

def build_llm(args) -> LLM:
    if args.llm_backend == "dummy":
        return DummyLLM(sleep_ms=args.dummy_llm_ms)
    if args.llm_backend == "transformers":
        if not args.llm_model:
            raise ValueError("--llm-model required for --llm-backend transformers")
        return TransformersLLM(
            model_name=args.llm_model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    raise ValueError(f"Unknown llm_backend: {args.llm_backend}")


# -----------------------
# Query source
# -----------------------
def load_queries(args) -> List[str]:
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            qs = [ln.strip() for ln in f if ln.strip()]
        if not qs:
            raise ValueError("queries_file is empty")
        return qs[: args.num_queries]
    # default: synthetic text queries
    return [f"example query {i}" for i in range(args.num_queries)]


# -----------------------
# RAG pipeline benchmark
# -----------------------
def run_benchmark(args):
    np.random.seed(args.seed)
    
    # Load index
    t0 = now()
    index = load_index(args)
    t1 = now()
    print(f"Index load time: {(t1 - t0)*1000:.1f} ms")

    # FAISS tuning
    if args.faiss_threads > 0:
        faiss.omp_set_num_threads(args.faiss_threads)
        print(f"FAISS threads: {args.faiss_threads}")
    if hasattr(index, "nprobe") and args.nprobe is not None:
        index.nprobe = args.nprobe
        print(f"Index nprobe: {args.nprobe}")

    # Build components
    # embedder = build_embedder(args)
    doc_store = build_doc_store(args)
    llm = build_llm(args)
    # queries = load_queries(args)
    queries = np.load(args.queries_file)

    # Timers per stage
    t_embed_ms: List[float] = []
    t_search_ms: List[float] = []
    t_fetch_ms: List[float] = []
    t_prompt_ms: List[float] = []
    t_llm_ms: List[float] = []
    t_total_ms: List[float] = []

    # # Warmup (small)
    # warm_q = queries[: min(5, len(queries))]
    # warm_vec = embedder.embed(warm_q)
    # _ = index.search(warm_vec, args.top_k)
    # if doc_store:
    #     _ = doc_store.get_many([0, 1, 2])
    # _ = llm.generate(["Warmup prompt"])

    # Main loop
    for q in queries:
        tA = now()

        # 1) Embed
        t0 = now()
        # qv = embedder.embed([q])  # (1, dim)
        t1 = now()

        # 2) Search
        D, I = index.search(qv, args.top_k)
        t2 = now()

        # 3) Fetch
        docs: List[str] = []
        if doc_store is not None:
            row_ids = [int(x) for x in I[0].tolist() if int(x) >= 0]
            docs = doc_store.get_many(row_ids)
        t3 = now()

        # 4) Prompt build
        # keep it simple; you can change template later
        context = "\n\n".join(docs[: args.max_context_docs])
        prompt = (
            "You are a helpful assistant.\n\n"
            f"Question: {q}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
        t4 = now()

        # 5) LLM inference
        _ = llm.generate([prompt])
        t5 = now()

        # Record timings
        t_embed_ms.append((t1 - t0) * 1000.0)
        t_search_ms.append((t2 - t1) * 1000.0)
        t_fetch_ms.append((t3 - t2) * 1000.0)
        t_prompt_ms.append((t4 - t3) * 1000.0)
        t_llm_ms.append((t5 - t4) * 1000.0)
        t_total_ms.append((t5 - tA) * 1000.0)

    print("\n--- Results ---")
    print_stats("embed", t_embed_ms)
    print_stats("search", t_search_ms)
    print_stats("fetch", t_fetch_ms)
    print_stats("prompt", t_prompt_ms)
    print_stats("llm", t_llm_ms)
    print_stats("total", t_total_ms)


def main():
    p = argparse.ArgumentParser(description="Local full RAG benchmark: embed -> FAISS -> doc fetch -> prompt -> LLM")

    # Index loading
    p.add_argument("--index-source", choices=["disk", "s3"], default="disk")
    p.add_argument("--index-path", required=True, help="Local index path (also download target if index-source=s3)")
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--force-download", action="store_true")

    p.add_argument("--s3-bucket", default=None)
    p.add_argument("--s3-index-key", default=None)
    p.add_argument("--s3-region", default="us-west-2")

    # Doc store options
    p.add_argument("--doc-store", choices=["none", "memory_jsonl", "sqlite"], default="sqlite")
    p.add_argument("--docs-jsonl-path", default=None, help="docs_by_row_*.jsonl")
    p.add_argument("--docs-db-path", default=None, help="docs_by_row_*.db (SQLite)")

    # Embedding options
    p.add_argument("--embed-backend", choices=["random", "sentence_transformers"], default="random")
    p.add_argument("--embed-model", default=None, help="e.g. sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--cosine", action="store_true", help="Normalize embeddings for cosine/IP")

    # FAISS tuning
    p.add_argument("--faiss-threads", type=int, default=0)
    p.add_argument("--nprobe", type=int, default=None)

    # LLM options
    p.add_argument("--llm-backend", choices=["dummy", "transformers"], default="dummy")
    p.add_argument("--dummy-llm-ms", type=float, default=50.0, help="Used if llm-backend=dummy")
    p.add_argument("--llm-model", default=None, help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)

    # Queries / run config
    p.add_argument("--batch-size", type=int, nargs='+', required=False, default=[32], help="List of batch sizes for querying")
    p.add_argument("--queries", type=str, required=False, default="triviaqa/triviaqa_encodings.npy", help="Path to the NumPy file containing embeddings")
    p.add_argument("--retrieved-docs", type=int, nargs='+', required=False, default=[5], help="List of numbers of docs retrieved per query")
    p.add_argument("--num-threads", type=int, nargs='+', required=False, default=[1], help="List of numbers of threads to run retrieval")
    p.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory where the results will be saved")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
