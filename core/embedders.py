# embedders.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import time

import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------------
# Base interface
# -------------------------------
class BaseEmbedder(ABC):
    @abstractmethod
    def embed_queries(self, queries: List[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...


# -------------------------------
# Synthetic embedder (benchmarking)
# -------------------------------
class SyntheticEmbedder(BaseEmbedder):
    """
    Synthetic embedder for benchmarking.

    - Deterministic: same text -> same vector
    - Optional sleep: simulate latency
    - Configurable dim to match FAISS index dim
    """
    def __init__(
        self,
        dim: int,
        *,
        normalize: bool = True,
        sleep_seconds: float = 0.0,
        seed: int = 0,
        query_prefix: str = "query: ",
        doc_prefix: str = "passage: ",
    ):
        self.dim = int(dim)
        self.normalize = bool(normalize)
        self.sleep_seconds = float(sleep_seconds)
        self.seed = int(seed)
        self.query_prefix = str(query_prefix)
        self.doc_prefix = str(doc_prefix)

    def _vec_for_text(self, t: str) -> np.ndarray:
        # stable 32/64-bit seed from text
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        text_seed = int.from_bytes(h, "little", signed=False) ^ self.seed

        rng = np.random.default_rng(text_seed)
        v = rng.standard_normal(self.dim, dtype=np.float32)

        if self.normalize:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        return v.astype(np.float32)

    def _embed(self, texts: List[str], prefix: str) -> np.ndarray:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        vecs = np.stack([self._vec_for_text(prefix + t) for t in texts], axis=0)
        return vecs.astype(np.float32)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self._embed(queries, prefix=self.query_prefix)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._embed(texts, prefix=self.doc_prefix)


# -------------------------------
# DPR embedder (NQ)
# -------------------------------
class DPREmbedderNQ(BaseEmbedder):
    """
    DPR encoders for NQ.
    IMPORTANT: DPR uses dot-product (inner product) and MUST NOT normalize.
    Compatible with facebook/wiki_dpr config: psgs_w100.nq.*
    """
    def __init__(
        self,
        device: str = "cpu",
        # normalize: bool = False,
        passage_format: str = "title_sep_text",
    ):
        """
        passage_format:
          - "title_sep_text": expects document strings already formatted like "title [SEP] text"
          - "raw": expects raw strings as-is
        """
        self.device = device
        # self.normalize = bool(normalize)  # keep configurable, but default False for DPR
        self.passage_format = passage_format
        self.question_encoder = None
        self.question_encoder = self._load_question_encoder()
        self.passage_encoder = None
       
    def _load_passage_encoder(self):
        if self.passage_encoder is None:
            self.passage_encoder = SentenceTransformer(
                "facebook-dpr-ctx_encoder-single-nq-base",
                device=self.device,
            )

    def _load_question_encoder(self):
        if self.question_encoder is None:
            self.question_encoder = SentenceTransformer(
                "facebook-dpr-question_encoder-single-nq-base",
                device=self.device,
            )

    def embed_queries(self, queries: List[str]) -> np.ndarray:

        self._load_question_encoder()
        return self.question_encoder.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=False,  # default False
        ).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        # If caller already formats passages as "title [SEP] text", pass them through.
        # If you want to enforce formatting here, do it upstream where you still have title/text fields.
        self._load_passage_encoder()
        return self.passage_encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,  # default False
        ).astype(np.float32)


# -------------------------------
# Optional: EmbeddingGemma (cosine style)
# -------------------------------
class GemmaEmbedder(BaseEmbedder):
    """
    EmbeddingGemma is typically used with cosine similarity:
      - normalize embeddings
      - use IndexFlatIP (dot-product on normalized vectors == cosine)
    """
    def __init__(self, device: str = "cpu", model_name: str = "google/embeddinggemma-300m"):
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


# -------------------------------
# Factory
# -------------------------------
def load_embedder(
    embedder_name: str,
    device: str,
    max_length: int = 512,  # reserved for future HF embedders
    *,
    delay_for_synthetic: float = 0.0,
    dim_for_synthetic: int = 768,
) -> BaseEmbedder:
    """
    embedder_name options (examples):
      - "synthetic"
      - "dpr_nq" or "dpr_qa"
      - "gemma"
    """
    name = (embedder_name or "").lower().strip()

    if name.startswith("synthetic"):
        return SyntheticEmbedder(
            dim=dim_for_synthetic,
            normalize=True,
            sleep_seconds=delay_for_synthetic,
        )

    # DPR (NQ) — dot-product geometry, do NOT normalize
    if name in {"dpr_nq", "dpr_qa", "dpr"}:
        return DPREmbedderNQ(device=device)

    # EmbeddingGemma — cosine workflow (normalize)
    if name in {"gemma", "embeddinggemma", "gemma_300m"}:
        return GemmaEmbedder(device=device)

    raise ValueError(f"Unknown embedder_name: {embedder_name!r}")
