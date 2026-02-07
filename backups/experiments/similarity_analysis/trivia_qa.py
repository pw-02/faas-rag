"""
compute_threshold_curve_triviaqa.py

Purpose:
- Load TriviaQA (rc.nocontext) questions
- Embed questions
- Compute nearest-neighbor cosine similarity
- Save threshold-curve data to CSV

Install:
  pip install datasets sentence-transformers faiss-cpu numpy pandas

Run:
  python compute_threshold_curve_triviaqa.py

Output:
  results/triviaqa_similarity_threshold_curve.csv
"""

import os
import numpy as np
import pandas as pd

from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter


# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "dataset_name": "mandarjoshi/trivia_qa",
    "dataset_config": "rc.nocontext",
    "split": "train",        # "train" is largest; "validation" also works

    "max_queries": None,     # e.g., 50_000 for speed
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 128,
    "thresholds": [0.70, 0.75, 0.80, 0.85, 0.90],
    "out_dir": "results",
}


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    # 1) Load TriviaQA
    print("Loading TriviaQA (rc.nocontext)…")
    ds = load_dataset(CONFIG["dataset_name"], CONFIG["dataset_config"])
    split = ds[CONFIG["split"]]

    if "question" not in split.column_names:
        raise KeyError(f"'question' field not found. Columns: {split.column_names}")

    queries = split["question"]
    
    dup_rate = 1 - len(set(queries)) / len(queries)
    print("Exact duplicate rate:", dup_rate)

    if CONFIG["max_queries"] is not None:
        queries = queries[: CONFIG["max_queries"]]

    n = len(queries)
    print(f"Loaded {n:,} questions.")

    # 2) Embed questions
    print("Encoding questions…")
    model = SentenceTransformer(CONFIG["embed_model"])
    emb = model.encode(
        queries,
        batch_size=CONFIG["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    # 3) Nearest-neighbor similarity
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    print("Computing nearest-neighbor similarities…")
    sims, _ = index.search(emb, 2)  # self + nearest neighbor
    nn_sim = sims[:, 1]

    # 4) Threshold curve
    thresholds = CONFIG["thresholds"]
    rates = [(nn_sim >= t).mean() for t in thresholds]

    # 5) Save CSV
    df = pd.DataFrame({
        "threshold": thresholds,
        "fraction": rates,
    })

    out_path = os.path.join(
        CONFIG["out_dir"],
        "triviaqa_similarity_threshold_curve.csv",
    )
    df.to_csv(out_path, index=False)

    print("\nSaved threshold curve to:", out_path)
    print(df)


if __name__ == "__main__":
    main()
