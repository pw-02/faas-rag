"""
compute_threshold_curve.py

Purpose:
- Load MS MARCO queries
- Embed queries
- Compute nearest-neighbor cosine similarity
- Save threshold-curve data to CSV

Install:
  pip install datasets sentence-transformers faiss-cpu numpy pandas

Run:
  python compute_threshold_curve.py

Output:
  results/threshold_curve.csv
"""

import os
import numpy as np
import pandas as pd

from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
CONFIG = {
    "msmarco_config": "v1.1",
    "split": "train",
    "max_queries": None,   # set None for all
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

    # 1) Load queries
    print("Loading MS MARCO queries…")
    ds = load_dataset("ms_marco", CONFIG["msmarco_config"])
    split = ds[CONFIG["split"]]

    if "query" not in split.column_names:
        raise KeyError(f"'query' field not found. Columns: {split.column_names}")

    queries = split["query"]
    if CONFIG["max_queries"] is not None:
        queries = queries[: CONFIG["max_queries"]]

    n = len(queries)
    print(f"Loaded {n:,} queries.")

    # 2) Embed queries
    print("Encoding queries…")
    model = SentenceTransformer(CONFIG["embed_model"])
    emb = model.encode(
        queries,
        batch_size=CONFIG["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,  # so inner product = cosine similarity
    ).astype("float32")

    # 3) Nearest-neighbor search
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    print("Computing nearest-neighbor similarities…")
    sims, idxs = index.search(emb, 2)  # self + nearest neighbor
    nn_sim = sims[:, 1]

    # 4) Compute threshold curve
    thresholds = CONFIG["thresholds"]
    rates = [(nn_sim >= t).mean() for t in thresholds]

    # 5) Save CSV
    df = pd.DataFrame({
        "threshold": thresholds,
        "fraction": rates,
    })

    out_path = os.path.join(CONFIG["out_dir"], "ms_macro_similarity_threshold_curve.csv")
    df.to_csv(out_path, index=False)

    print("\nSaved threshold curve to:", out_path)
    print(df)


if __name__ == "__main__":
    main()
