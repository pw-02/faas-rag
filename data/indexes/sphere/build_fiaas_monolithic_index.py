#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3

import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset

# Fixed number of vectors to add per batch.
NUM_VECTORS_PER_BATCH = 100_000

# Mapping of allowed index sizes to dataset names.
DATASET_MAPPING = {
    "100k": "mohdumar/SPHERE_100K",
    "100m": "mohdumar/SPHERE_100M",
    "899m": "mohdumar/SPHERE_899M",
}

def create_faiss_index_from_vectors(vectors: np.ndarray, dim: int) -> faiss.Index:
    """
    Create and populate a FAISS IVF+ScalarQuantizer index from a dense matrix of vectors.
    Uses cosine similarity via Inner Product + L2 normalization.
    """
    total_vectors = vectors.shape[0]

    # Heuristic for IVF lists
    C = 1
    nlists = max(1, C * int(np.sqrt(total_vectors)))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        dim,
        nlists,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT,
    )

    # Train on a subset (cap for huge datasets)
    # train_size = min(max(100_000, total_vectors // 10), 1_000_000)
    # train_size = min(train_size, total_vectors)
    train_size = int(total_vectors / 10)

    print(f"Training FAISS index with {train_size} vectors (nlists={nlists})...")
    index.train(vectors[:train_size])

    print("Adding dataset vectors to the FAISS index in batches...")
    for i in tqdm(range(0, total_vectors, NUM_VECTORS_PER_BATCH), desc="Batches Processed", unit="batch"):
        batch = vectors[i : i + NUM_VECTORS_PER_BATCH]
        index.add(batch)

    return index

def write_docs_jsonl_by_row(dataset, out_path: str, text_field: str):
    """
    Writes docs as JSONL keyed by FAISS row_id (insertion order).
    """
    print(f"Writing row-aligned docs JSONL to {out_path} (text field: {text_field})")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row_id, ex in enumerate(tqdm(dataset, desc="docs.jsonl")):
            f.write(json.dumps({"row_id": row_id, "text": ex[text_field]}) + "\n")

def write_docs_sqlite_by_row(dataset, out_path: str, text_field: str):
    """
    Writes docs to SQLite keyed by FAISS row_id (insertion order).
    """
    print(f"Writing row-aligned docs SQLite DB to {out_path} (text field: {text_field})")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    conn = sqlite3.connect(out_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            row_id INTEGER PRIMARY KEY,
            text   TEXT
        )
    """)

    # Speed up bulk inserts
    conn.execute("BEGIN")
    for row_id, ex in enumerate(tqdm(dataset, desc="docs.db")):
        cur.execute(
            "INSERT OR REPLACE INTO docs (row_id, text) VALUES (?, ?)",
            (row_id, ex[text_field]),
        )
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index + row-aligned doc artifacts from HF dataset")
    parser.add_argument("--index-size", type=str, default="100k", help="100k, 100m, 899m")
    parser.add_argument("--output-dir", type=str, default="data/indexes/sphere")
    parser.add_argument("--dim", type=int, default=768)

    # IMPORTANT: this builder loads vectors into RAM; streaming not supported in this script.
    parser.add_argument("--dataset-streaming", action="store_true",
                        help="Not supported here (this script needs full vectors in memory).")

    # dataset field names
    parser.add_argument("--text-field", type=str, default="raw",
                        help="Dataset field containing raw chunk text.")

    # doc outputs
    parser.add_argument("--write-docs", choices=["none", "jsonl", "sqlite", "both"], default="both")
    args = parser.parse_args()

    if args.dataset_streaming:
        raise ValueError("dataset-streaming is not supported in this builder (it loads all vectors into RAM).")

    index_size = args.index_size.lower()
    if index_size not in DATASET_MAPPING:
        raise ValueError("Invalid index size. Choose one of: 100k, 100m, 899m")

    dataset_name = DATASET_MAPPING[index_size]

    # --- Load dataset (non-streaming) ---
    print(f"Loading Hugging Face dataset: {dataset_name} ...")
    dataset = load_dataset(dataset_name, split="train", streaming=False)

    cols = dataset.column_names
    if "vector" not in cols:
        raise ValueError("Dataset does not contain a 'vector' column.")
    if args.text_field not in cols:
        raise ValueError(f"Dataset does not contain text field '{args.text_field}'. Available columns: {cols}")

    # Load vectors into memory
    vectors = np.array(dataset["vector"], dtype="float32")
    total_vectors = vectors.shape[0]
    print(f"Dataset loaded. Total vectors: {total_vectors}")

    if vectors.ndim != 2 or vectors.shape[1] != args.dim:
        raise ValueError(f"Vector dim mismatch: expected (*, {args.dim}), got {vectors.shape}")

    # --- Build FAISS index ---
    index = create_faiss_index_from_vectors(vectors, args.dim)

    # --- Write artifacts ---
    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, f"index_cc_monolithic_{index_size}.faiss")
    docs_jsonl_path = os.path.join(args.output_dir, f"cc_docs_{index_size}.jsonl")
    docs_db_path = os.path.join(args.output_dir, f"cc_docs_{index_size}.db")

    print(f"Saving FAISS index -> {index_path}")
    faiss.write_index(index, index_path)

    # IMPORTANT: row-aligned doc store must match vector insertion order.
    # Since we added vectors in the same order as dataset["vector"], we must iterate the dataset in that same order.
    if args.write_docs in ("jsonl", "both"):
        write_docs_jsonl_by_row(dataset, docs_jsonl_path, args.text_field)

    if args.write_docs in ("sqlite", "both"):
        write_docs_sqlite_by_row(dataset, docs_db_path, args.text_field)

    print("Done.")

if __name__ == "__main__":
    main()
