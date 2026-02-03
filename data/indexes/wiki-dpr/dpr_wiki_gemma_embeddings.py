import os
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# -------- Config --------
DATASET = "tomaztc/wiki_dpr_gemma_embeddings"
SPLIT = "train"
STREAMING = True

N_VECTORS = 100_000          # <-- how many passages/vectors to index
OUT_DIR = "faiss_wiki_subset"
INDEX_TYPE = "flat_ip"       # "flat_ip" or "ivf_ip"
N_LISTS = 4096               # used only for IVF
BATCH_SIZE = 8192            # how many vectors to buffer before adding
STORE_TEXT = False           # set True if you want text saved (big!)
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Load streaming dataset
ds = load_dataset(DATASET,
                  name="psgs_w100.multiset.compressed",
                  split=SPLIT, 
                  streaming=STREAMING)

# Peek first row to get embedding dim
first = next(iter(ds))
dim = len(first["embedding"])

# Re-create iterator (since we consumed one element)
ds = load_dataset(DATASET, split=SPLIT, streaming=STREAMING)

# Import faiss
try:
    import faiss  # pip install faiss-cpu   (or faiss-gpu)
except ImportError as e:
    raise SystemExit(
        "FAISS not installed. Try: pip install faiss-cpu  (or faiss-gpu)"
    ) from e

# Choose index
if INDEX_TYPE == "flat_ip":
    # exact search, cosine-friendly if vectors are normalized
    index = faiss.IndexFlatIP(dim)
elif INDEX_TYPE == "ivf_ip":
    # faster for large N, approximate
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, N_LISTS, faiss.METRIC_INNER_PRODUCT)
else:
    raise ValueError("INDEX_TYPE must be 'flat_ip' or 'ivf_ip'")

meta_path = os.path.join(OUT_DIR, "meta.jsonl")
index_path = os.path.join(OUT_DIR, "index.faiss")

# Buffers
buf_vecs = []
buf_meta = []
added = 0

# If IVF, we need training vectors first
TRAIN_FOR_IVF = 100_000 if INDEX_TYPE == "ivf_ip" else 0
train_vecs = []

with open(meta_path, "w", encoding="utf-8") as mf:
    for row in tqdm(ds, total=N_VECTORS, desc="Streaming"):
        # stop when we have enough vectors
        if added >= N_VECTORS:
            break

        v = np.asarray(row["embedding"], dtype=np.float32)

        # Optional: normalize for cosine similarity via inner product
        # (only do this if you want cosine retrieval)
        n = np.linalg.norm(v) + 1e-12
        v = v / n

        # Collect training vecs for IVF
        if INDEX_TYPE == "ivf_ip" and len(train_vecs) < TRAIN_FOR_IVF:
            train_vecs.append(v)

        # Buffer for add()
        buf_vecs.append(v)
        meta = {
            "id": row.get("id"),
            "title": row.get("title", ""),
        }
        if STORE_TEXT:
            meta["text"] = row.get("text", "")
        buf_meta.append(meta)

        # Flush batch
        if len(buf_vecs) >= BATCH_SIZE:
            X = np.vstack(buf_vecs).astype(np.float32)

            # Train IVF once
            if INDEX_TYPE == "ivf_ip" and not index.is_trained:
                if len(train_vecs) < min(TRAIN_FOR_IVF, N_LISTS * 20):
                    # heuristic: want at least ~20 vectors per list if possible
                    print(
                        f"Warning: only {len(train_vecs)} training vectors collected; "
                        f"IVF may be suboptimal."
                    )
                T = np.vstack(train_vecs).astype(np.float32)
                index.train(T)

            index.add(X)

            for m in buf_meta:
                mf.write(json.dumps(m, ensure_ascii=False) + "\n")

            added += len(buf_vecs)
            buf_vecs.clear()
            buf_meta.clear()

# Flush remaining
if buf_vecs:
    X = np.vstack(buf_vecs).astype(np.float32)
    if INDEX_TYPE == "ivf_ip" and not index.is_trained:
        T = np.vstack(train_vecs).astype(np.float32)
        index.train(T)
    index.add(X)

    with open(meta_path, "a", encoding="utf-8") as mf:
        for m in buf_meta:
            mf.write(json.dumps(m, ensure_ascii=False) + "\n")

    added += len(buf_vecs)

# Save index
faiss.write_index(index, index_path)

print(f"\nDone.")
print(f"Vectors indexed: {index.ntotal} (target was {N_VECTORS})")
print(f"Index saved to: {index_path}")
print(f"Metadata saved to: {meta_path}")
print(f"Dim: {dim}, Index type: {INDEX_TYPE}")
