import numpy as np
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def l2(a, b):
    return float(np.linalg.norm(a - b))

def show_pair(name, a, b):
    return f"{name:28s} cosine={cosine(a,b):.6f}  l2={l2(a,b):.6f}"

USE_MULTISET = False
DATASET_NAME = "psgs_w100.multiset.no_index" if USE_MULTISET else "psgs_w100.nq.no_index"
ST_MODEL_NAME = "facebook-dpr-ctx_encoder-multiset-base" if USE_MULTISET else "facebook-dpr-ctx_encoder-single-nq-base"
T_MODEL_NAME = "facebook/dpr-ctx_encoder-multiset-base" if USE_MULTISET else "facebook/dpr-ctx_encoder-single-nq-base"


# -------------------------------
# Dataset (NQ)
# -------------------------------
ds = load_dataset(
    "facebook/wiki_dpr",
    name=DATASET_NAME,
    split="train",
    streaming=True,
)

# -------------------------------
# Models
# -------------------------------
tf_model_name = T_MODEL_NAME
tok = DPRContextEncoderTokenizer.from_pretrained(tf_model_name)
enc = DPRContextEncoder.from_pretrained(tf_model_name).eval()

st_model_name = ST_MODEL_NAME
st = SentenceTransformer(st_model_name)

# -------------------------------
# Compare first N samples
# -------------------------------
N = 5

for idx, row in enumerate(ds):
    if idx >= N:
        break

    target = np.asarray(row["embeddings"], dtype=np.float32)
    title, text = row["title"], row["text"]

    with torch.no_grad():
        inputs = tok(title, text, return_tensors="pt", truncation=True)
        v_tf = enc(**inputs).pooler_output[0].cpu().numpy().astype(np.float32)

    v_st = st.encode(
        [f"{title} [SEP] {text}"],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0].astype(np.float32)

    print(f"\n=== Sample {idx} ===")
    print("norms:",
          f"target={np.linalg.norm(target):.3f}",
          f"tf={np.linalg.norm(v_tf):.3f}",
          f"st={np.linalg.norm(v_st):.3f}")

    print(show_pair("TF DPRContext vs dataset", v_tf, target))
    print(show_pair("ST wrapper vs dataset",   v_st, target))
    print(show_pair("ST wrapper vs TF DPR",    v_st, v_tf))
