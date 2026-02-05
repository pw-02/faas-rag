import threading
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from faasrag.core.args import (
    EmbedderConfig,
    DPREmbedderConfig,
    SyntheticEmbedderConfig,
    GemmaEmbedderConfig,
)


# -------------------------
# DPR Embedder
# -------------------------

class DPREmbedder:
    """
    DPR-style dual encoder.
    Uses dot-product retrieval => embeddings are NOT normalized.
    """

    def __init__(
        self,
        device: str,
        query_encoder_id: str,
        passage_encoder_id: str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        self.name = "dpr"
        self.device = device
        self.query_encoder_id = query_encoder_id
        self.passage_encoder_id = passage_encoder_id
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self._query_model: Optional[SentenceTransformer] = None
        self._passage_model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()

    def _load_query_model(self) -> SentenceTransformer:
        if self._query_model is None:
            with self._lock:
                if self._query_model is None:
                    model = SentenceTransformer(self.query_encoder_id, device=self.device)
                    model.eval()
                    self._query_model = model
        return self._query_model

    def _load_passage_model(self) -> SentenceTransformer:
        if self._passage_model is None:
            with self._lock:
                if self._passage_model is None:
                    model = SentenceTransformer(self.passage_encoder_id, device=self.device)
                    model.eval()
                    self._passage_model = model
        return self._passage_model

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        model = self._load_query_model()
        emb = model.encode(
            queries,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=self.show_progress_bar,
        )
        return emb.astype(np.float32, copy=False)

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        model = self._load_passage_model()
        emb = model.encode(
            passages,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=self.show_progress_bar,
        )
        return emb.astype(np.float32, copy=False)


# -------------------------
# Synthetic Embedder
# -------------------------

class SyntheticEmbedder:
    def __init__(
        self,
        *,
        dim: int,
        sleep_seconds: float,
        query_prefix: str,
        passage_prefix: str,
        normalize: bool,
    ):
        import hashlib, time

        self.dim = int(dim)
        self.sleep_seconds = float(sleep_seconds)
        self.query_prefix = str(query_prefix)
        self.passage_prefix = str(passage_prefix)
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
        return np.stack([self._vec(self.query_prefix + q) for q in queries])

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        if self.sleep_seconds:
            self._time.sleep(self.sleep_seconds)
        return np.stack([self._vec(self.passage_prefix + t) for t in passages])


# -------------------------
# Gemma Embedder
# -------------------------

class GemmaEmbedder:
    def __init__(self, *, device: str, model_name: str):
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        return self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_passages(self, passages: List[str]) -> np.ndarray:
        return self.model.encode(
            passages,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


# -------------------------
# Builder
# -------------------------

Embedder = Union[DPREmbedder, SyntheticEmbedder, GemmaEmbedder]


def build_embedder(cfg: EmbedderConfig, device: str) -> Embedder:
    if cfg.type == "dpr":
        c: DPREmbedderConfig = cfg
        return DPREmbedder(
            device=device,
            query_encoder_id=c.query_encoder_id,
            passage_encoder_id=c.passage_encoder_id,
            batch_size=c.batch_size,
        )

    if cfg.type == "synthetic":
        c: SyntheticEmbedderConfig = cfg
        return SyntheticEmbedder(
            dim=c.dim,
            sleep_seconds=c.sleep_time,
            query_prefix=c.query_prefix,
            passage_prefix=c.passage_prefix,
            normalize=c.normalize,
        )

    if cfg.type == "gemma":
        c: GemmaEmbedderConfig = cfg
        return GemmaEmbedder(
            device=device,
            model_name=c.model_name,
        )

    raise ValueError(f"Unknown embedder type: {cfg.type!r}")
