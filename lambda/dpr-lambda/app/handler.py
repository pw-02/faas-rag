import json
import os
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# -----------------------------
# Config
# -----------------------------
HERE = os.path.dirname(__file__)

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(HERE, "models"))
MODEL_FILE = os.environ.get("MODEL_FILE", "model.onnx")  # e.g. model.int8.onnx
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))     # must match ONNX export if fixed

# Hugging Face caches must be writable on Lambda -> /tmp
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)

# Optional: reduce tokenizer parallel warnings/noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------
# Load tokenizer + ONNX session
# -----------------------------
_tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR, use_fast=True)

sess_options = ort.SessionOptions()
# sess_options.intra_op_num_threads = 1  # optional tuning

_session = ort.InferenceSession(
    os.path.join(MODELS_DIR, MODEL_FILE),
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

# Discover model IO once (prevents "Invalid Output Name" / unexpected inputs)
_INPUT_NAMES = {i.name for i in _session.get_inputs()}
_OUTPUT_OBJS = _session.get_outputs()
_OUTPUT_NAMES = [o.name for o in _OUTPUT_OBJS]
_DEFAULT_OUTPUT_NAME = _OUTPUT_OBJS[0].name if _OUTPUT_OBJS else None

print("ONNX inputs:", sorted(_INPUT_NAMES))
print("ONNX outputs:", _OUTPUT_NAMES)
for i in _session.get_inputs():
    print("ONNX input spec:", i.name, i.shape, i.type)
for o in _session.get_outputs():
    print("ONNX output spec:", o.name, o.shape, o.type)


# -----------------------------
# Helpers
# -----------------------------
def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    last_hidden_state: (batch, seq, hidden)
    attention_mask:    (batch, seq)
    returns pooled:    (batch, hidden)
    """
    mask = attention_mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)  # (batch, seq, 1)
    masked = last_hidden_state * mask
    summed = masked.sum(axis=1)  # (batch, hidden)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)  # (batch, 1)
    return summed / counts


def _embed(text: str) -> np.ndarray:
    enc = _tokenizer(
        text,
        return_tensors="np",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    inputs: Dict[str, np.ndarray] = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }

    # Only pass token_type_ids if the ONNX model expects it
    if "token_type_ids" in _INPUT_NAMES:
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(enc["input_ids"])
        inputs["token_type_ids"] = token_type_ids.astype(np.int64)

    # Run inference. Use the first output by default; avoids hardcoding "embedding"
    outputs = _session.run(None, inputs)
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs")

    out0 = outputs[0]

    # Common cases:
    # 1) Already pooled embedding: (batch, hidden) -> return (hidden,)
    if out0.ndim == 2:
        return out0[0]

    # 2) Token embeddings/last_hidden_state: (batch, seq, hidden) -> mean pool -> (hidden,)
    if out0.ndim == 3:
        pooled = _mean_pool(out0, inputs["attention_mask"])
        return pooled[0]

    raise RuntimeError(f"Unexpected ONNX output shape: {out0.shape}")


def _parse_body(event: Any) -> Dict[str, Any]:
    """
    Supports:
    - API Gateway/Lambda proxy: {"body": "...json..."} or {"body": {..}}
    - Direct invoke with dict: {"query": "..."}
    """
    body = event.get("body", event) if isinstance(event, dict) else event
    if isinstance(body, str):
        body = body.strip()
        if body:
            return json.loads(body)
        return {}
    if isinstance(body, dict):
        return body
    return {}


# -----------------------------
# Lambda handler
# -----------------------------
def handler(event, context):
    body = _parse_body(event)

    query = body.get("query") or body.get("text") or body.get("q")
    if not query:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": "Missing 'query' (or 'text'/'q') field"}),
        }

    try:
        vec = _embed(str(query)).astype(np.float32).tolist()
    except Exception as e:
        # Include minimal debug info; avoid dumping sensitive input
        return {
            "statusCode": 500,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(
                {
                    "error": "Embedding failed",
                    "detail": str(e),
                    "onnx_outputs": _OUTPUT_NAMES,
                    "onnx_inputs": sorted(_INPUT_NAMES),
                }
            ),
        }

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"embedding": vec, "dim": len(vec)}),
    }