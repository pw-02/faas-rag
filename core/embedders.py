# embedders.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import hashlib
import time

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_queries(self, texts: List[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...


class SyntheticEmbedder(BaseEmbedder):
    """
    Synthetic embedder for benchmarking.

    - Deterministic: same text -> same vector (stable retrieval)
    - Optional sleep: simulate embed latency
    - Configurable dim to match your FAISS index dim
    """
    def __init__(
        self,
        dim: int,
        *,
        normalize: bool = True,
        sleep_seconds: float = 0.0,
        seed: int = 0,
    ):
        self.dim = int(dim)
        self.normalize = normalize
        self.sleep_seconds = float(sleep_seconds)
        self.seed = int(seed)

    def _vec_for_text(self, t: str) -> np.ndarray:
        # Stable 32-bit seed from text
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        text_seed = int.from_bytes(h, "little", signed=False) ^ self.seed

        rng = np.random.default_rng(text_seed)
        v = rng.standard_normal(self.dim, dtype=np.float32)

        if self.normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v

    def _embed(self, texts: List[str], prefix: str = "") -> np.ndarray:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        vecs = np.stack([self._vec_for_text(prefix + t) for t in texts], axis=0)
        return vecs.astype("float32")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        # optional: mimic BGE behavior with a different prefix for queries
        return self._embed(texts, prefix="query: ")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, prefix="passage: ")




class HuggingFaceEmbedder(BaseEmbedder):
    """
    Generic HF encoder embedder with masked mean pooling.
    """
    def __init__(self, model_name: str,
                  device: str, 
                  max_length: int = 512,
                normalize: bool = True):
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.detach().cpu().numpy().astype("float32")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts)


# IMPORTANT:
    # - For BGE: embed_queries will apply "query:" prefix internally.
    # - For generic encoders: it just embeds the raw query.
class BGEEmbedder(HuggingFaceEmbedder):
    """
    BGE-specific embedder that applies instruction prefixes:
      - queries
      - passages/documents
    """
    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return super()._embed([f"query: {t}" for t in texts])

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return super()._embed([f"passage: {t}" for t in texts])
    

class DPREmbedder(BaseEmbedder):
    """
    DPR embedder using DPR encoders' pooler_output.

    - Uses DPRQuestionEncoder / DPRContextEncoder under the hood (via AutoModel)
    - Uses pooler_output (not mean pooling)
    - Applies "query:" and "passage:" prefixes by default (can be disabled)
    """
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int = 512,
        normalize: bool = True,
        *,
        add_prefix: bool = True,
    ):
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self.add_prefix = add_prefix

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Decide which prefix behavior to use based on model name
        lname = model_name.lower()
        self._is_ctx = ("ctx_encoder" in lname) or ("context" in lname)
        self._is_q = ("question_encoder" in lname) or ("query" in lname)

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # DPR returns pooler_output; AutoModel should expose it for DPR encoders.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            # Fallback: CLS token (should be rare for DPR, but keeps it robust)
            pooled = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.detach().cpu().numpy().astype("float32")

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        # If user loaded the context encoder, we still can embed queries,
        # but it will usually perform worse. We don't block it.
        if self.add_prefix:
            texts = [f"query: {t}" for t in texts]
        return self._encode(texts)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.add_prefix:
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)



def load_embedder(
    embedder_name: str,
    device: str,
    max_length: int = 512,
    *,
    sleep_seconds_for_synthetic: float = 0.0,
    dim_for_synthetic: int = 768,
) -> BaseEmbedder:
    name = embedder_name.lower()

    if name.startswith("synthetic"):
        return SyntheticEmbedder(
            dim=dim_for_synthetic,
            normalize=True,
            sleep_seconds=sleep_seconds_for_synthetic,
        )

    # DPR (must come before generic HF)
    if "dpr" in name or "ctx_encoder" in name or "question_encoder" in name:
        return DPREmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    # BGE
    if "bge" in name:
        return BGEEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    # E5 (you likely want prefixes too; consider adding an E5Embedder similar to BGE)
    if "e5" in name:
        return HuggingFaceEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    # # SentenceTransformers models: you *can* keep using HuggingFaceEmbedder,
    # # but quality may differ from SentenceTransformer pooling.
    # # For best results, you'd add a SentenceTransformerEmbedder separately.
    # if "sentence-transformers" in name:
    #     return HuggingFaceEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    return HuggingFaceEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)
