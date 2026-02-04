# faasrag/core/builders.py
from __future__ import annotations

from typing import Any

from faasrag.core.args import (
    EmbedderConfig,
    DPRNQEmbedderConfig,
    DPRMultiSetEmbedderConfig,
    SyntheticEmbedderConfig,
    GemmaEmbedderConfig,
    LlamaGeneratorConfig,
)
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class DPREmbedderNQ:
    def __init__(self, device: str, passage_encoder: str, question_encoder: str):
        self.passage_model = SentenceTransformer(passage_encoder, device=device)
        self.question_model = SentenceTransformer(question_encoder, device=device)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        # DPR: DO NOT normalize
        return self.question_model.encode(
            queries, convert_to_numpy=True, normalize_embeddings=False
        ).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        # DPR: DO NOT normalize
        return self.passage_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=False
        ).astype(np.float32)


class DPREmbedderMultiset(DPREmbedderNQ):
    pass


class SyntheticEmbedder:
    def __init__(self, *, dim: int, sleep_seconds: float, query_prefix: str, doc_prefix: str, normalize: bool):
        import hashlib, time
        self.dim = int(dim)
        self.sleep_seconds = float(sleep_seconds)
        self.query_prefix = str(query_prefix)
        self.doc_prefix = str(doc_prefix)
        self.normalize = bool(normalize)
        self._hash = hashlib.blake2b
        self._time = time

    def _vec(self, s: str) -> np.ndarray:
        h = self._hash(s.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(h, "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim, dtype=np.float32)
        if self.normalize:
            v /= (np.linalg.norm(v) + 1e-12)
        return v

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        if self.sleep_seconds:
            self._time.sleep(self.sleep_seconds)
        return np.stack([self._vec(self.query_prefix + q) for q in queries]).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.sleep_seconds:
            self._time.sleep(self.sleep_seconds)
        return np.stack([self._vec(self.doc_prefix + t) for t in texts]).astype(np.float32)


class GemmaEmbedder:
    def __init__(self, *, device: str, model_name: str):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.model.encode(
            queries, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)



def build_embedder(cfg: EmbedderConfig, *, device: str):
    """
    Create a runtime embedder instance from EmbedderConfig.
    """
    t = cfg.type
    
    if t == "dpr_nq":
        c: DPRNQEmbedderConfig | None = cfg.dpr_nq
        if c is None:
            raise ValueError("EmbedderConfig.type is dpr_nq but dpr_nq config is None")

        # DPR IMPORTANT: normalize must be False for dot-product geometry
        if c.normalize:
            raise ValueError("DPR NQ embedder must have normalize=false")

        return DPREmbedderNQ(
            device=device,
            passage_encoder=c.passage_encoder,
            question_encoder=c.question_encoder,
            # metric/dim are usually validated elsewhere
        )

    if t == "dpr_multiset":
        c: DPRMultiSetEmbedderConfig | None = cfg.dpr_multiset
        if c is None:
            raise ValueError("EmbedderConfig.type is dpr_multiset but dpr_multiset config is None")
        if c.normalize:
            raise ValueError("DPR multiset embedder must have normalize=false")

        return DPREmbedderMultiset(
            device=device,
            passage_encoder=c.passage_encoder,
            question_encoder=c.question_encoder,
        )

    if t == "synthetic":
        c: SyntheticEmbedderConfig | None = cfg.synthetic
        if c is None:
            raise ValueError("EmbedderConfig.type is synthetic but synthetic config is None")

        return SyntheticEmbedder(
            dim=c.dim,
            sleep_seconds=c.sleep_time,
            query_prefix=c.query_prefix,
            doc_prefix=c.passage_prefix,
            normalize=True,
        )

    if t == "gemma":
        c: GemmaEmbedderConfig | None = cfg.gemma
        if c is None:
            raise ValueError("EmbedderConfig.type is gemma but gemma config is None")

        return GemmaEmbedder(
            device=device,
            model_name=c.model_name,
            # Gemma is usually cosine-style => normalize=True inside embedder
        )

    raise ValueError(f"Unknown embedder type: {t!r}")
# build_generator