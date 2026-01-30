# embedders.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_queries(self, texts: List[str]) -> np.ndarray:
        ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        ...


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Generic HF encoder embedder with masked mean pooling.
    """
    def __init__(self, model_name: str, device: str, max_length: int = 512, normalize: bool = True):
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


def load_embedder(embedder_name: str, device: str, max_length: int = 512) -> BaseEmbedder:
    """
    Factory returning the right embedder implementation.

    - BGE: uses query/passage prefixes via BGEEmbedder
    - E5 / sentence-transformers: generic HF embedder is usually fine
    - DPR: requires a dual-encoder wrapper (not included here yet)
    """
    name = embedder_name.lower()

    if "bge" in name:
        return BGEEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    if "e5" in name or "sentence-transformers" in name:
        return HuggingFaceEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)

    if "dpr" in name:
        raise NotImplementedError(
            "DPR requires separate ctx and question encoders. Add a DPREmbedder to support it."
        )

    # Fallback: try generic HF embedder for experimentation
    return HuggingFaceEmbedder(embedder_name, device=device, max_length=max_length, normalize=True)
