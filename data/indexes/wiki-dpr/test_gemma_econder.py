import csv
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# -------------------------------
# Utilities
# -------------------------------
def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def l2(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def norm(a):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32)))

# -------------------------------
# Model
# -------------------------------
MODEL_NAME = "google/embeddinggemma-300m"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(MODEL_NAME, device=device)
model.eval()

@torch.no_grad()
def embed_docs(texts, normalize=False):
    """
    For EmbeddingGemma, prefer encode_document() if available.
    Fallback: model.encode() with an explicit document prompt.
    """
    if hasattr(model, "encode_document"):
        emb = model.encode_document(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    else:
        # fallback prompt approach
        emb = model.encode(
            [f"document: {t}" for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    return emb.astype(np.float32)

# -------------------------------
# Dataset (streaming)
# -------------------------------
ds = load_dataset(
    "tomaztc/wiki_dpr_gemma_embeddings",
    split="train",
    streaming=True,
)

print(ds)
# -------------------------------
# Run comparison
# -------------------------------
NUM_SAMPLES = 5
OUT_CSV = "wiki_dpr_gemma_check.csv"

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "idx",
        "title_snippet",
        "text_len",
        "target_dim",
        "target_norm",
        "gemma_dim",
        "gemma_norm",
        "gemma_cosine",
        "gemma_l2",
    ])

    for idx, sample in enumerate(ds):
        if idx >= NUM_SAMPLES:
            break

        title = sample.get("title", "")
        text = sample["text"]

        target = np.asarray(sample["embedding"], dtype=np.float32)
        print(np.linalg.norm(target))
    
        v = embed_docs([text], normalize=True)[0]

        # sanity check: same dimensionality
        if v.shape[0] != target.shape[0]:
            raise ValueError(
                f"Dim mismatch at idx={idx}: gemma={v.shape[0]} vs target={target.shape[0]}"
            )

        writer.writerow([
            idx,
            title[:80].replace("\n", " "),
            len(text),
            int(target.shape[0]),
            norm(target),
            int(v.shape[0]),
            norm(v),
            cosine(v, target),
            l2(v, target),
        ])

print(f"\nSaved results to: {OUT_CSV}")
print(f"Device used: {device}")
