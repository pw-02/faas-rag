"""
compute_threshold_curve_beir_quora.py

Purpose:
- Load BEIR Quora queries via ir_datasets
- Embed queries
- Compute nearest-neighbor cosine similarity
- Save threshold-curve data to CSV

Install:
  pip install ir_datasets sentence-transformers faiss-cpu numpy pandas

Run:
  python compute_threshold_curve_beir_quora.py

Output:
  results/beir_quora_similarity_threshold_curve.csv
"""

import os
import numpy as np
import pandas as pd

import ir_datasets
import faiss
from sentence_transformers import SentenceTransformer


CONFIG = {
    "irds_name": "beir/quora",
    "max_queries": None,   # e.g., 50_000 for speed

    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 128,
    "thresholds": [0.70, 0.75, 0.80, 0.85, 0.90],
    "out_dir": "results",
}


def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    print(f"Loading BEIR Quora queries from ir_datasets: {CONFIG['irds_name']} …")
    dataset = ir_datasets.load(CONFIG["irds_name"])

    queries = []
    for q in dataset.queries_iter():
        queries.append(q.text)
        if CONFIG["max_queries"] is not None and len(queries) >= CONFIG["max_queries"]:
            break

    n = len(queries)
    if n == 0:
        raise RuntimeError("Loaded 0 queries. Check dataset name.")

    print(f"Loaded {n:,} queries.")

    print("Encoding queries…")
    model = SentenceTransformer(CONFIG["embed_model"])
    emb = model.encode(
        queries,
        batch_size=CONFIG["batch_size"],
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    print("Computing nearest-neighbor similarities…")
    sims, _ = index.search(emb, 2)  # self + nearest neighbor
    nn_sim = sims[:, 1]

    thresholds = CONFIG["thresholds"]
    rates = [(nn_sim >= t).mean() for t in thresholds]

    df = pd.DataFrame({"threshold": thresholds, "fraction": rates})
    out_path = os.path.join(
        CONFIG["out_dir"],
        "beir_quora_similarity_threshold_curve.csv",
    )
    df.to_csv(out_path, index=False)

    print("\nSaved threshold curve to:", out_path)
    print(df)


if __name__ == "__main__":
    main()
