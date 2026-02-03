#!/usr/bin/env python3
"""
Build an IVF (IVF-Flat or IVF-PQ) FAISS index over facebook/wiki_dpr (psgs_w100)
using DPR context embeddings. Requires non-streaming dataset (needs training pass).

Outputs:
  - index.faiss
  - meta.jsonl (row-aligned)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--index_type", type=str, required=True, choices=["ivfflat", "ivfpq"])
    p.add_argument("--max_passages", type=int, default=500_000)

    p.add_argument("--train_size", type=int, default=200_000)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--nprobe", type=int, default=16)

    p.add_argument("--m", type=int, default=64)
    p.add_argument("--nbits", type=int, default=8)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model_name", type=str, default="facebook/dpr-ctx_encoder-single-nq-base")
    p.add_argument("--normalize", action="store_true")
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


def build_ivf_index(dim: int, args: argparse.Namespace) -> faiss.Index:
    quantizer = faiss.IndexFlatIP(dim)
    if args.index_type == "ivfflat":
        idx = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        if dim % args.m != 0:
            raise ValueError(f"IVF-PQ requires dim % m == 0. Got dim={dim}, m={args.m}.")
        idx = faiss.IndexIVFPQ(quantizer, dim, args.nlist, args.m, args.nbits, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = args.nprobe
    return idx


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"[info] device={device}, index_type={args.index_type}")

    print("[info] loading dataset facebook/wiki_dpr (psgs_w100) [non-streaming]")
    ds = load_dataset("facebook/wiki_dpr", "psgs_w100", split="train", streaming=False)

    print(f"[info] loading model {args.model_name}")
    tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model_name)
    model = DPRContextEncoder.from_pretrained(args.model_name).to(device).eval()

    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    meta_file = open(meta_path, "w", encoding="utf-8")

    # ---- 1) Training vectors ----
    train_target = min(args.train_size, args.max_passages, len(ds))
    print(f"[info] collecting training vectors train_size={train_target}")

    train_vecs_list: List[np.ndarray] = []
    train_metas: List[dict] = []
    batch_texts: List[str] = []
    batch_metas: List[dict] = []

    for i in range(train_target):
        ex = ds[i]
        batch_texts.append(ex["text"])
        batch_metas.append({"id": ex["id"], "title": ex["title"], "text": ex["text"]})

        if len(batch_texts) < args.batch_size:
            continue

        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)
        train_vecs_list.append(vecs)
        train_metas.extend(batch_metas)
        batch_texts.clear()
        batch_metas.clear()

    if batch_texts:
        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)
        train_vecs_list.append(vecs)
        train_metas.extend(batch_metas)
        batch_texts.clear()
        batch_metas.clear()

    train_vecs = np.vstack(train_vecs_list)
    dim = train_vecs.shape[1]
    print(f"[info] got {train_vecs.shape[0]} training vecs (dim={dim})")

    # ---- 2) Build + train IVF index ----
    index = build_ivf_index(dim, args)
    print("[info] training IVF index...")
    index.train(train_vecs)
    print("[info] training complete")

    # Add training vecs first (and write metadata)
    index.add(train_vecs)
    for m in train_metas:
        meta_file.write(json.dumps(m, ensure_ascii=False) + "\n")

    indexed = train_vecs.shape[0]
    print(f"[info] added training vecs; indexed={indexed}")

    # ---- 3) Add remaining vectors ----
    end_i = min(args.max_passages, len(ds))
    batch_texts = []
    batch_metas = []

    for i in range(train_target, end_i):
        ex = ds[i]
        batch_texts.append(ex["text"])
        batch_metas.append({"id": ex["id"], "title": ex["title"], "text": ex["text"]})

        if len(batch_texts) < args.batch_size:
            continue

        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)
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

    if batch_texts:
        vecs = embed_texts(batch_texts, tokenizer, model, device, args.max_length)
        if args.normalize:
            faiss.normalize_L2(vecs)
        index.add(vecs)
        for m in batch_metas:
            meta_file.write(json.dumps(m, ensure_ascii=False) + "\n")
        indexed += len(batch_texts)
        batch_texts.clear()
        batch_metas.clear()

    meta_file.close()

    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))
    print(f"[done] indexed={indexed}, wrote index.faiss and meta.jsonl")
    print(f"[info] IVF nlist={index.nlist} nprobe={index.nprobe}")


if __name__ == "__main__":
    main()
