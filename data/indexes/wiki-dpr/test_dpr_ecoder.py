import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer
import onnxruntime as ort

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

# ---- ONNX config ----
ONNX_DIR = "lambda/dpr-lambda/app/models"  # contains model.onnx + tokenizer files
ONNX_MODEL_FILE = "model.onnx"
MAX_LENGTH = 509

# Dataset
ds = load_dataset(
    "facebook/wiki_dpr",
    name=DATASET_NAME,
    split="train",
    streaming=True,
)

# Transformers DPR context encoder
tok = DPRContextEncoderTokenizer.from_pretrained(T_MODEL_NAME)
enc = DPRContextEncoder.from_pretrained(T_MODEL_NAME).eval()

# SentenceTransformers DPR context encoder
st = SentenceTransformer(ST_MODEL_NAME)

# ONNX model + tokenizer
onnx_tok = AutoTokenizer.from_pretrained(ONNX_DIR, use_fast=True)
sess = ort.InferenceSession(
    os.path.join(ONNX_DIR, ONNX_MODEL_FILE),
    providers=["CPUExecutionProvider"],
)
ONNX_OUTPUT_NAME = sess.get_outputs()[0].name

def embed_onnx(text: str) -> np.ndarray:
    e = onnx_tok(
        text,
        return_tensors="np",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    tti = e.get("token_type_ids")
    if tti is None:
        tti = np.zeros_like(e["input_ids"])

    inputs = {
        "input_ids": e["input_ids"].astype(np.int64),
        "attention_mask": e["attention_mask"].astype(np.int64),
        "token_type_ids": tti.astype(np.int64),
    }
    (vec,) = sess.run([ONNX_OUTPUT_NAME], inputs)
    return vec[0].astype(np.float32)

# Compare first N samples
N = 5

for idx, row in enumerate(ds):
    if idx >= N:
        break

    target = np.asarray(row["embeddings"], dtype=np.float32)
    title, text = row["title"], row["text"]

    with torch.no_grad():
        inputs = tok(title, text, return_tensors="pt", truncation=True)
        v_tf = enc(**inputs).pooler_output[0].cpu().numpy().astype(np.float32)

    formatted = f"{title} [SEP] {text}"

    v_st = st.encode(
        [formatted],
        convert_to_numpy=True,
        normalize_embeddings=False,
    )[0].astype(np.float32)

    v_onnx = embed_onnx(formatted)

    print(f"\n=== Sample {idx} ===")
    print("norms:",
          f"target={np.linalg.norm(target):.3f}",
          f"tf={np.linalg.norm(v_tf):.3f}",
          f"st={np.linalg.norm(v_st):.3f}",
          f"onnx={np.linalg.norm(v_onnx):.3f}")

    print(show_pair("TF DPRContext vs dataset", v_tf, target))
    print(show_pair("ST wrapper vs dataset",   v_st, target))
    print(show_pair("ONNX vs dataset",         v_onnx, target))
    print(show_pair("ONNX vs ST",              v_onnx, v_st))
    print(show_pair("ONNX vs TF DPR",          v_onnx, v_tf))
