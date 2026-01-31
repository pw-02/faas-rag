from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
import math
import time
from contextlib import contextmanager

import torch
import faiss
from tqdm import tqdm

from core.embedders import load_embedder
from core.generators import load_generator, GenerationConfig
from core.docstores import load_docstore, BaseDocStore


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


@contextmanager
def timer(timings: Dict[str, float], key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = time.perf_counter() - t0


class RagPipelineBase:
    """
    Shared RAG pipeline scaffolding. Subclasses can override:
      - pre_faiss_hook(...)
      - post_embed_hook(...)
      - maybe retrieve_docs(...) if you later want real retrieval objects
    """

    def __init__(
        self,
        *,
        generator_name: str,
        embedder_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        vector_index_path: str,
        docstore_path: str = "",
        docstore_type: str = "jsonl",
        docstore: Optional[BaseDocStore] = None,
        top_k: int = 5,
        max_context_docs: Optional[int] = None,
        max_new_tokens: int = 256,
        embedder_max_length: int = 512,
        do_sample: bool = False,
        sleep_seconds: float = 0.0,
        n_probe: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = int(top_k)
        self.max_context_docs = int(max_context_docs) if max_context_docs is not None else int(top_k)
        self.show_progress = bool(show_progress)
        self.batch_size = max(1, int(batch_size))

        # --- Embedder ---
        self.embedder = load_embedder(
            embedder_name=embedder_name,
            device=self.device,
            max_length=embedder_max_length,
        )

        # --- Generator ---
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
            self.docstore = load_docstore(docstore_path=docstore_path, docstore_type=docstore_type)

        self._check_faiss_dim()

    def _check_faiss_dim(self) -> None:
        test = self.embedder.embed_queries(["dimension check"])  # (1, D)
        d_model = int(test.shape[1])
        d_index = int(self.index.d)
        if d_model != d_index:
            raise ValueError(
                f"Embedding dim ({d_model}) != FAISS index dim ({d_index}). "
                f"Your index was built with a different embedder (or different settings)."
            )

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
        raw = self.generator.generate(prompt)
        return raw.split("Answer:", 1)[1].strip() if "Answer:" in raw else raw.strip()

    # ---- hooks: subclasses can override these to add caching etc. ----
    def post_embed_hook(self, q_vecs, batch: List[str]) -> None:
        return None

    def pre_faiss_hook(self, q_vecs, batch: List[str]):
        return q_vecs

    def run(
        self,
        queries: Union[str, List[str]],
        *,
        return_prompt: bool = False,
        return_contexts: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        single = isinstance(queries, str)
        queries_list = [queries] if single else list(queries)
        limit = len(queries_list)
        if limit == 0:
            return {} if single else []

        k = self.top_k
        starts = range(0, limit, self.batch_size)
        if self.show_progress:
            starts = tqdm(starts, total=math.ceil(limit / self.batch_size), desc="RAG batches")

        batch_results: List[Dict[str, Any]] = []

        for start in starts:
            batch = queries_list[start : start + self.batch_size]
            if not batch:
                continue

            timings_batch: Dict[str, float] = {}

            # embed
            t0 = time.perf_counter()
            q_vecs = self.embedder.embed_queries(batch)
            timings_batch["embed_s"] = time.perf_counter() - t0

            self.post_embed_hook(q_vecs, batch)
            q_vecs = self.pre_faiss_hook(q_vecs, batch)

            # faiss
            t0 = time.perf_counter()
            distances, indices = self.index.search(q_vecs, k)
            timings_batch["faiss_s"] = time.perf_counter() - t0

            # docstore + prompt + generate timings (as you had)
            docstore_s = 0.0
            prompt_s = 0.0
            generate_s = 0.0
            keep = min(self.max_context_docs, k)

            for i, q in enumerate(batch):
                t0 = time.perf_counter()
                for idx in indices[i][:keep]:
                    _ = self.docstore.get(int(idx))
                docstore_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                _ = self.build_prompt(q, [])
                prompt_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                _ = self.generate("")
                generate_s += time.perf_counter() - t0

            timings_batch["docstore_s"] = docstore_s
            timings_batch["prompt_s"] = prompt_s
            timings_batch["generate_s"] = generate_s
            timings_batch["total_s"] = sum(
                timings_batch[x] for x in ("embed_s", "faiss_s", "docstore_s", "prompt_s", "generate_s")
            )

            batch_results.append(
                {
                    "batch_start": start,
                    "batch_size": len(batch),
                    "top_k": k,
                    "max_context_docs": self.max_context_docs,
                    "timings": timings_batch,
                }
            )

        return batch_results[0] if single else batch_results
