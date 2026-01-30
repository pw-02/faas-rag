from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
import json
import time
import csv
from contextlib import contextmanager

import torch
import faiss

from embedders import load_embedder
from generators import load_generator, GenerationConfig
from docstores import load_docstore, BaseDocStore, Doc


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
class RagPipeline:
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
        top_k: int = 5,
        max_context_docs: int = 5,
        max_new_tokens: int = 256,
        embedder_max_length: int = 512,
        do_sample: bool = False,
        sleep_seconds: float = 0.0,  # for synthetic generator only
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        self.max_context_docs = max_context_docs

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

    # def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDoc]:
    #     """
    #     Retrieval only: embed query -> FAISS -> docstore fetch.
    #     """
    #     k = top_k or self.top_k

    #     # IMPORTANT:
    #     # - For BGE: embed_queries will apply "query:" prefix internally.
    #     # - For generic encoders: it just embeds the raw query.
    #     q_vec = self.embedder.embed_queries([query])  # (1, D) float32 on CPU

    #     distances, indices = self.index.search(q_vec, k)

    #     docs: List[RetrievedDoc] = []
    #     for rank, idx in enumerate(indices[0].tolist()):
    #         doc = self.docstore.get(idx)  # Doc(id, text, meta)
    #         docs.append(
    #             RetrievedDoc(
    #                 doc_id=str(doc.id),
    #                 text=str(doc.text),
    #                 score=float(distances[0][rank]),
    #             )
    #         )

    #     return docs[: self.max_context_docs]

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

    # -----------------------------
    # Profiling + CSV export
    # -----------------------------
    def _save_results_csv(self, results: List[Dict[str, Any]], path: str) -> None:
        """
        Save flattened results + stage timings to CSV.
        Lists (doc ids/scores) are stored as JSON strings.
        """
        rows: List[Dict[str, Any]] = []
        for r in results:
            t = r.get("timings", {})
            rows.append(
                {
                    # "query": r.get("query", ""),
                    # "answer": r.get("answer", ""),
                    "n_retrieved": r.get("n_retrieved", 0),
                    "embed_s": t.get("embed_s", 0.0),
                    "faiss_s": t.get("faiss_s", 0.0),
                    "docstore_s": t.get("docstore_s", 0.0),
                    "prompt_s": t.get("prompt_s", 0.0),
                    "generate_s": t.get("generate_s", 0.0),
                    "total_s": t.get("total_s", 0.0),
                    "top_doc_ids": json.dumps(r.get("top_doc_ids", [])),
                    "top_scores": json.dumps(r.get("top_scores", [])),
                }
            )

        fieldnames = list(rows[0].keys()) if rows else [
            # "query",
            # "answer",
            
            "num_retrieved_docs",
            "query_embed_time_s",
            "index_search_time_s",
            "docstore_fetch_time_s",
            "prompt_build_time_s",
            "generation_time_s",
            "end_to_end_time_s",

            "retrieved_doc_ids",
            "retrieved_doc_scores",
        ]


        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # -----------------------------
    # Public API
    # -----------------------------
    def run(
        self,
        queries: Union[str, List[str]],
        *,
        csv_path: Optional[str] = None,
        return_prompt: bool = False,
        return_contexts: bool = True,
        top_k: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Run RAG for one query or a batch, with per-stage profiling.

        Args:
            queries: str or list[str]
            csv_path: if provided, saves results to CSV
            return_prompt: include final prompt in each result (debug)
            return_contexts: include retrieved docs in each result
            top_k: override retrieval top_k for this run

        Returns:
            dict for single query, list[dict] for batch
        """
        single = isinstance(queries, str)
        queries_list = [queries] if single else list(queries)

        results: List[Dict[str, Any]] = []

        for q in queries_list:
            timings: Dict[str, float] = {}

            # ---- Retrieval staged timing ----
            with timer(timings, "embed_s"):
                q_vec = self.embedder.embed_queries([q])  # (1, D)

            with timer(timings, "faiss_s"):
                k = top_k or self.top_k
                distances, indices = self.index.search(q_vec, k)

            with timer(timings, "docstore_s"):
                retrieved_docs: List[RetrievedDoc] = []
                for rank, idx in enumerate(indices[0].tolist()):
                    doc = self.docstore.get(idx)
                    retrieved_docs.append(
                        RetrievedDoc(
                            doc_id=str(doc.id),
                            text=str(doc.text),
                            score=float(distances[0][rank]),
                        )
                    )
                retrieved_docs = retrieved_docs[: self.max_context_docs]

            # ---- Prompt + generation ----
            with timer(timings, "prompt_s"):
                prompt = self.build_prompt(q, retrieved_docs)

            with timer(timings, "generate_s"):
                answer = self.generate(prompt)

            timings["total_s"] = sum(
                timings.get(k, 0.0)
                for k in ("embed_s", "faiss_s", "docstore_s", "prompt_s", "generate_s")
            )

            item: Dict[str, Any] = {
                "query": q,
                "answer": answer,
                "n_retrieved": len(retrieved_docs),
                "timings": timings,
                "top_doc_ids": [d.doc_id for d in retrieved_docs],
                "top_scores": [d.score for d in retrieved_docs],
            }

            if return_contexts:
                item["contexts"] = [
                    {"doc_id": d.doc_id, "score": d.score, "text": d.text}
                    for d in retrieved_docs
                ]

            if return_prompt:
                item["prompt"] = prompt

            results.append(item)

        if csv_path:
            self._save_results_csv(results, csv_path)

        return results[0] if single else results

    

if __name__ == "__main__":
    #example usage
  if __name__ == "__main__":
    # -----------------------------
    # Example usage: synthetic generator
    # -----------------------------
    pipeline = RagPipeline(
        generator_name="synthetic",                 # uses SyntheticAnswerGenerator
        embedder_name="BAAI/bge-base-en-v1.5",
        vector_index_path="data/indexes/synthetic/flat_ip_d768_n100000_norm1.index",
        docstore_path="data/indexes/sphere/cc_docs_100k.jsonl",
        docstore_type="jsonl",
        top_k=5,
        max_context_docs=3,
        max_new_tokens=128,
        sleep_seconds=0.1,                           # simulate LLM latency
    )
    queries = [
        "What is retrieval-augmented generation?",
        "What is FAISS used for?",
    ]

    results = pipeline.run(
        queries,
        csv_path="rag_profile_single_node.csv",                  # saves timings + outputs
        return_prompt=False,
        return_contexts=True,
    )
    # Pretty-print one result
    for r in results:
        print("=" * 80)
        print("QUERY:", r["query"])
        print("ANSWER:", r["answer"])
        print("TIMINGS:", r["timings"])
