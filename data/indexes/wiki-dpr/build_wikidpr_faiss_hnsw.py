#!/usr/bin/env python3
"""
Build an HNSW FAISS index over facebook/wiki_dpr (psgs_w100) using DPR context embeddings.

Outputs:
  - index.faiss
  - meta.jsonl  (row-aligned)
"""

import argparse
import json
import os
from typing import List, Optional

import faiss
import numpy as np
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

from datasets import load_dataset
ds = load_dataset("facebook/wiki_dpr", "psgs_w100", split="train")
print(ds[0])

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=False, default="data/indexes/wiki-dpr")
    p.add_argument("--max_passages", type=int, default=50_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model_name", type=str, default="facebook/dpr-ctx_encoder-single-nq-base")
    p.add_argument("--normalize", action="store_true")

    p.add_argument("--hnsw_m", type=int, default=32)
    p.add_argument("--ef_construction", type=int, default=200)
    p.add_argument("--ef_search", type=int, default=128)

    p.add_argument("--save_every", type=int, default=50_000)
    return p.parse_args()


def get_device(device_arg: Optional[str]) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts(
    texts: List[str],
    tokenizer: DPRContextEncoderTokenizerFast,
    model: DPRContextEncoder,
    device: str,
    max_length: int,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        vecs = model(**inputs).pooler_output

    return vecs.cpu().numpy().astype("float32")


def iter_examples(ds, streaming: bool, max_passages: int):
    if streaming:
        count = 0
        for ex in ds:
            if count >= max_passages:
                break
            yield ex
            count += 1
    else:
        n = min(max_passages, len(ds))
        for i in range(n):
            yield ds[i]


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[info] device={device}")

    print("[info] loading dataset facebook/wiki_dpr (psgs_w100)")
    ds = load_dataset("facebook/wiki_dpr", "psgs_w100", split="train", streaming=args.streaming)

    print(f"[info] loading model {args.model_name}")
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model_name)
    model = DPRContextEncoder.from_pretrained(args.model_name).to(device).eval()

    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    meta_file = open(meta_path, "w", encoding="utf-8")

    index = None
    indexed = 0

    batch_texts: List[str] = []
    batch_metas: List[dict] = []

    print("[info] building HNSW index")
    for ex in iter_examples(ds, args.streaming, args.max_passages):
        batch_texts.append(ex["text"])
        batch_metas.append({"id": ex["id"], "title": ex["title"], "text": ex["text"]})

        if len(batch_texts) < args.batch_size:
            continue

        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)

        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = args.ef_construction
            index.hnsw.efSearch = args.ef_search
            print(f"[info] initialized HNSW dim={dim}")

        index.add(vecs)

        for m in batch_metas:
            meta_file.write(json.dumps(m, ensure_ascii=False) + "\n")

        indexed += len(batch_texts)
        batch_texts.clear()
        batch_metas.clear()

        if indexed % 5000 == 0:
            print(f"[progress] indexed={indexed}")
        if indexed % args.save_every == 0:
            faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))
            print(f"[checkpoint] saved index at indexed={indexed}")

    # flush remainder
    if batch_texts:
        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)

        if index is None:
            dim = vecs.shape[1]
            index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = args.ef_construction
            index.hnsw.efSearch = args.ef_search
            print(f"[info] initialized HNSW dim={dim}")

        index.add(vecs)
        for m in batch_metas:
            meta_file.write(json.dumps(m, ensure_ascii=False) + "\n")
        indexed += len(batch_texts)
        batch_texts.clear()
        batch_metas.clear()

    meta_file.close()

    if index is None:
        raise RuntimeError("No passages indexed.")

    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))
    print(f"[done] indexed={indexed}, wrote index.faiss and meta.jsonl")


if __name__ == "__main__":
    main()
