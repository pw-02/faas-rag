from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import torch

from faasrag.core.args import (
    GeneratorConfig,
    EmbedderConfig,
    IndexConfig,
    CacheConfig,
    DocStoreConfig,
    Passage,
)
from faasrag.core.caches import build_cache
from faasrag.core.embedders import build_embedder
from faasrag.core.generators import build_generator
from faasrag.core.docstores import load_docstore
from faasrag.core.indexes import load_index
from faasrag.core.prompts import get_prompt_strategy, extract_short_answer


class RagPipeline:
    def __init__(
        self,
        *,
        generator_cfg: GeneratorConfig,
        embedder_cfg: EmbedderConfig,
        index_cfg: IndexConfig,
        docstore_cfg: DocStoreConfig,
        artifact_dir: str,
        prompt_type: str,
        max_ctx_chars: int = 4000,
        device: Optional[str] = None,
        cache_cfg: Optional[CacheConfig] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
        retrieve_only: bool = False,
        seed: Optional[int] = None,
    ):
        self.logger = logger or logging.getLogger("rag_pipeline")

        self.retrieve_only = bool(retrieve_only)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.top_k = int(top_k)
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")

        self.prompt_type = prompt_type
        self.max_ctx_chars = int(max_ctx_chars)
        self.prompt_fn = get_prompt_strategy(self.prompt_type)

        self.seed = seed

        self.cache = None
        self.generator = None

        # 1) Embedder
        self.embedder = build_embedder(embedder_cfg, device=self.device)

        # 2) Index
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)

        # 3) Dim sanity
        dim = self.sanity_check_dimensions()

        # 4) Docstore
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir)

        # 5) Cache
        if cache_cfg is not None:
            self.cache = build_cache(cache_cfg, dim=dim, seed=self.seed)

        # 6) Generator
        if not self.retrieve_only:
            self.generator = build_generator(generator_cfg, device=self.device)

        self.logger.info(
            "RagPipeline initialized prompt=%s retrieve_only=%s top_k=%d device=%s",
            self.prompt_type,
            self.retrieve_only,
            self.top_k,
            self.device,
        )

    def sanity_check_dimensions(self) -> int:
        test = self.embedder.embed_queries(["dim check"])
        embed_dim = int(test.shape[1])

        index_dim = getattr(self.index, "d", None)
        if index_dim is None:
            raise ValueError(f"Index object {type(self.index)} has no attribute `.d`")

        if embed_dim != int(index_dim):
            raise ValueError(f"Embed dim {embed_dim} != index dim {index_dim}")

        self.logger.info("Embedder dim %d matches index dim %d", embed_dim, int(index_dim))
        return embed_dim

    # ------------------------------------------------
    # Main entry point (used by gRPC)
    # ------------------------------------------------
    def run(self, query: str, top_k: Optional[int] = None, max_tokens: int = 0) -> dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        
        k = self.top_k if top_k is None else int(top_k)
        
        if k < 0:
            raise ValueError("top_k must be >= 0")
        
        # k == 0 => pure generation
        no_retrieval = k == 0
        passages: list[Passage] = []
        retrieved_doc_ids: list[str] = []
        retrieve_ms = 0.0

        # -------------------------
        # Retrieval
        # -------------------------
        if not no_retrieval:
            t0 = time.perf_counter()
            
            qvec = self.embedder.embed_queries([query])
            if hasattr(qvec, "detach"):
                qvec = qvec.detach().cpu().numpy()
            qvec = np.asarray(qvec, dtype=np.float32)

            if self.cache is not None:
                distances, indices = self.cache.cached_search(qvec, k=k, backend_index=self.index)
            else:
                distances, indices = self.index.search(qvec, k)

            for rank, pid in enumerate(indices[0]):
                if pid < 0:
                    continue
                d = self.docstore.get(str(pid))
                if d is None:
                    continue

                passages.append(
                    Passage(
                        pid=int(pid),
                        title=d.get("title", ""),
                        text=d.get("text", ""),
                        score=float(distances[0][rank]),
                    )
                )

            retrieved_doc_ids = [str(p.pid) for p in passages]
            retrieve_ms = (time.perf_counter() - t0) * 1000.0

        if self.retrieve_only or self.generator is None:
            return {
                "answer": "",
                "retrieved_doc_ids": retrieved_doc_ids,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "retrieve_ms": retrieve_ms,
                "decode_ms": 0.0,
            }

        # -------------------------
        # Prompt + generation
        # -------------------------
        if no_retrieval:
            messages = get_prompt_strategy("no_retrieval")(query)
        else:
            messages = self.prompt_fn(query, passages, self.max_ctx_chars)

        t1 = time.perf_counter()

        text, out_tokens = self.generator.generate_messages(messages)
        answer = extract_short_answer(text)

        decode_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "answer": answer,
            "retrieved_doc_ids": retrieved_doc_ids,
            "prompt_tokens": 0,
            "output_tokens": int(out_tokens),
            "retrieve_ms": retrieve_ms,
            "decode_ms": decode_ms,
        }
