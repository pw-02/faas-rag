from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Dict, Optional, Union, Any
import json
import time
import csv
from contextlib import contextmanager

import torch
import faiss

from core.embedders import load_embedder
from core.generators import load_generator, GenerationConfig
from core.docstores import load_docstore, BaseDocStore, Doc
from typing import Iterable
from tqdm import tqdm  # pip install tqdm

# -----------------------------
# Data models
# -----------------------------
@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float

# -----------------------------
# Tiny timing helper
# -----------------------------
@contextmanager
def timer(timings: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


# -----------------------------
# RAG Pipeline
# -----------------------------
class RagPipelineSingleNode:
    """
    Minimal RAG:
      1) embed query
      2) FAISS top-k
      3) load docs (DocStore)
      4) build prompt
      5) generate answer (Generator)

    Optional:
      - profile stage timings
      - save results to CSV
    """

    def __init__(
        self,
        generator_name: str,  # e.g. "distilgpt2" OR "synthetic"
        embedder_name: str = "BAAI/bge-base-en-v1.5",  # must match FAISS index
        device: Optional[str] = None,  # None = auto-detect
        vector_index_path: str = "",
        docstore_path: str = "",
        docstore_type: str = "jsonl",
        docstore: Optional[BaseDocStore] = None,
        top_k: int = 5, #number of docs to retrieve per query
        max_context_docs: int = None,  #number of docs to include in prompt
        max_new_tokens: int = 256,
        embedder_max_length: int = 512,
        do_sample: bool = False,
        sleep_seconds: float = 0.0,  # for synthetic generator only
        n_probe: Optional[int] = None,  # for IVF indexes
        show_progress: bool = True,
        batch_size: int = 16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.max_context_docs = max_context_docs if max_context_docs is not None else top_k
        self.show_progress = show_progress
        self.batch_size = max(1, int(batch_size))


        # --- Embedder (for vectors) ---
        self.embedder = load_embedder(
            embedder_name=embedder_name,
            device=self.device,
            max_length=embedder_max_length,
        )

        # --- Generator (for text) ---
        self.generator = load_generator(
            generator_name=generator_name,
            device=self.device,
            gen_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            ),
            sleep_seconds_for_synthetic=sleep_seconds,
        )

        # --- FAISS index ---
        if not vector_index_path:
            raise ValueError("vector_index_path is required")
        self.index = faiss.read_index(vector_index_path)

        if n_probe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = n_probe

        # --- Docstore ---
        if docstore is not None:
            self.docstore = docstore
        else:
            if not docstore_path:
                raise ValueError("docstore_path is required when docstore is not provided")
            self.docstore = load_docstore(
                docstore_path=docstore_path,
                docstore_type=docstore_type,
            )

        # Optional: sanity check dimension match
        self._check_faiss_dim()

    # -----------------------------
    # Retrieval
    # -----------------------------
    def _check_faiss_dim(self) -> None:
        test = self.embedder.embed_queries(["dimension check"])  # (1, D)
        d_model = int(test.shape[1])
        d_index = int(self.index.d)
        if d_model != d_index:
            raise ValueError(
                f"Embedding dim ({d_model}) != FAISS index dim ({d_index}). "
                f"Your index was built with a different embedder (or different settings)."
            )
        
    # -----------------------------
    # Prompting + Generation
    # -----------------------------
    def build_prompt(self, query: str, docs: List[RetrievedDoc]) -> str:
        context_blocks = [f"[{i}] {d.text}" for i, d in enumerate(docs, start=1)]
        context = "\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"
        return (
            "You are a helpful assistant. Answer the question using the provided context.\n"
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

    def generate(self, prompt: str) -> str:
        """
        Calls the generator abstraction. Returns only the completion after 'Answer:' when present.
        """
        raw = self.generator.generate(prompt)
        return raw.split("Answer:", 1)[1].strip() if "Answer:" in raw else raw.strip()

    
    def run(self,queries: Union[str, List[str]],*,
            return_prompt: bool = False,      # kept for compatibility (unused)
            return_contexts: bool = True,     # kept for compatibility (unused)
            ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        
        single = isinstance(queries, str)
        queries_list = [queries] if single else list(queries)
        limit = len(queries_list)
        if limit == 0:
            return {} if single else []

        k = self.top_k
        starts = range(0, limit, self.batch_size)
        if self.show_progress:
            starts = tqdm(starts,total=math.ceil(limit / self.batch_size),desc="RAG batches",)

        batch_results: List[Dict[str, Any]] = []
        for start in starts:
            batch = queries_list[start : start + self.batch_size]
            if not batch:
                continue

            timings_batch: Dict[str, float] = {}

            # ---- Embed (batch) ----
            t0 = time.perf_counter()
            q_vecs = self.embedder.embed_queries(batch)
            timings_batch["embed_s"] = time.perf_counter() - t0

            # ---- FAISS (batch) ----
            t0 = time.perf_counter()
            distances, indices = self.index.search(q_vecs, k)
            timings_batch["faiss_s"] = time.perf_counter() - t0

            # ---- Docstore + prompt + generate (summed per batch) ----
            docstore_s = 0.0
            prompt_s = 0.0
            generate_s = 0.0

            keep = min(self.max_context_docs, k)
            
            for i, q in enumerate(batch):
                # docstore
                t0 = time.perf_counter()
                for idx in indices[i][:keep]:
                    _ = self.docstore.get(int(idx))
                docstore_s += time.perf_counter() - t0

                # prompt
                t0 = time.perf_counter()
                _ = self.build_prompt(q, [])
                prompt_s += time.perf_counter() - t0

                # generate
                t0 = time.perf_counter()
                _ = self.generate("")
                generate_s += time.perf_counter() - t0

            timings_batch["docstore_s"] = docstore_s
            timings_batch["prompt_s"] = prompt_s
            timings_batch["generate_s"] = generate_s
            timings_batch["total_s"] = sum(
                timings_batch[x]
                for x in ("embed_s", "faiss_s", "docstore_s", "prompt_s", "generate_s")
            )

            batch_results.append({
                "batch_start": start,
                "batch_size": len(batch),
                "top_k": k,
                "max_context_docs": self.max_context_docs,
                "timings": timings_batch,
            })

        return batch_results[0] if single else batch_results
