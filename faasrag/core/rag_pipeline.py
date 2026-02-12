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
from contextlib import contextmanager


@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


class RagPipeline:
    def __init__(
        self,
        *,
        generator_cfg: GeneratorConfig,
        embedder_cfg: EmbedderConfig,
        index_cfg: IndexConfig,
        docstore_cfg: DocStoreConfig,
        docstore_backend: str,
        artifact_dir: str,
        prompt_type: str,
        max_ctx_chars: int = 4000,
        cache_cfg: Optional[CacheConfig] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
        retrieve_only: bool = False,
        seed: Optional[int] = None,
    ):
        self.logger = logger or logging.getLogger("rag_service")

        self.retrieve_only = bool(retrieve_only)
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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
        self.embedder = build_embedder(embedder_cfg)

        # 2) Index
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)

        # 3) Dim sanity
        dim = self.sanity_check_dimensions()

        # 4) Docstore
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir, backend=docstore_backend)

        # 5) Cache
        if cache_cfg is not None:
            self.cache = build_cache(cache_cfg, dim=dim, seed=self.seed)

        # 6) Generator
        if not self.retrieve_only:
            self.generator = build_generator(generator_cfg)

        self.logger.info(
            "RagPipeline initialized prompt=%s retrieve_only=%s top_k=%d embedder_device=%s generator_device=%s",
            self.prompt_type,
            self.retrieve_only,
            self.top_k,
            getattr(self.embedder, "device", "unknown"),
            getattr(self.generator, "device", "unknown"),
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
    def run(self, query: str) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must be non-empty")
        
        no_retrieval = (self.top_k == 0) or (self.prompt_type == "no_retrieval")
        passages: list[Passage] = []
        retrieved_doc_ids: list[str] = []
        timings: dict[str, float] = {}
        cache_used = False
        cache_hits = 0
        cache_misses = 0

        # -------------------------
        # Retrieval
        # -------------------------
        if no_retrieval:
            timings.update({"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
        else:     
            with timed(timings, "embed_s"):
                qvec = self.embedder.embed_queries([query])
                if hasattr(qvec, "detach"):
                    qvec = qvec.detach().cpu().numpy()
                qvec = np.asarray(qvec, dtype=np.float32)

            cache_stats: dict[str, Any] | None = None
            with timed(timings, "ann_s"):
                if self.cache is not None:
                    cache_used = True
                    distances, indices, cache_stats = self.cache.cached_search(
                        qvec, k=self.top_k, backend_index=self.index
                    )
                else:
                    distances, indices = self.index.search(qvec, self.top_k)
            
            if cache_stats:
                cache_hits = int(cache_stats.get("hits", 0))
                cache_misses = int(cache_stats.get("misses", 0))
            
            with timed(timings, "docstore_s"):
                for rank, pid in enumerate(indices[0]):
                    if pid < 0:
                        continue

                    doc = self.docstore.get(str(pid))
                    if not doc:
                        continue
                    passages.append(
                        Passage(
                            pid=int(pid),
                            title=doc.get("title", ""),
                            text=doc.get("text", ""),
                            score=float(distances[0][rank]),
                        )
                    )

                retrieved_doc_ids = [str(p.pid) for p in passages]

        # -------------------------
        # Early exit (retrieve-only)
        # -------------------------
        if self.retrieve_only or self.generator is None:
            timings.update({"prompt_s": 0.0, "decode_s": 0.0})
            return {
                "answer": "",
                "retrieved_doc_ids": retrieved_doc_ids,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "timings_s": timings,
                "cache_used": cache_used,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }
        # -------------------------
        # Prompt construction
        # -------------------------
        with timed(timings, "prompt_s"):
            if no_retrieval:
                messages = get_prompt_strategy("no_retrieval")(query)
            else:
                messages = self.prompt_fn(query, passages, self.max_ctx_chars)
        
        # -------------------------
        # Generation
        # -------------------------
        with timed(timings, "decode_s"):
            answer, prompt_tokens, completion_tokens, total_tokens = self.generator.generate_messages(messages)
        
        return {
        # "answer": answer,
        "answer": extract_short_answer(answer, max_chars=10),
        "retrieved_doc_ids": retrieved_doc_ids,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "timings_s": timings,
        "cache_used": cache_used,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
    }

