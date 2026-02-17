from __future__ import annotations
import logging
import time
import re
import math
import collections
from typing import Any, Optional

import numpy as np
from datetime import datetime, timezone
from contextlib import contextmanager

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
from faasrag.core.prompts import PromptBuildMethodType, build_rag_messages, build_stage1_scoring_messages
from faasrag.core.utils import append_csv_row


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
        prompt_build_method: str,
        max_ctx_chars: int = 4000,
        cache_cfg: Optional[CacheConfig] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
        retrieve_only: bool = False,
        seed: Optional[int] = None,
        always_log_results: bool = False,
    ):
        self.logger = logger or logging.getLogger("rag_service")

        self.retrieve_only = bool(retrieve_only)
        self.always_log_results = bool(always_log_results)
        self.top_k = int(top_k)
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        elif self.top_k == 0 or self.retrieve_only is True:
            self.logger.warning(
                "No retrieval will be performed, pipeline will rely entirely on the generator's prior knowledge."
            )
        else:
            self.logger.info("top_k=%d: Retrieval will be performed with up to top_k passages included in the prompt.", self.top_k)

        if prompt_build_method.upper() == "QA_STRICT":
            self.prompt_build_method = PromptBuildMethodType.QA_STRICT
        elif prompt_build_method.upper() == "QA_OPEN":
            self.prompt_build_method = PromptBuildMethodType.QA_OPEN
        elif prompt_build_method.upper() == "FEW_SHOT":
            self.prompt_build_method = PromptBuildMethodType.FEW_SHOT
        elif prompt_build_method.upper() == "LLM_ONLY":
            self.prompt_build_method = PromptBuildMethodType.LLM_ONLY
        elif prompt_build_method.upper() == "LOGIT_RAG_STAGE1":
            self.prompt_build_method = PromptBuildMethodType.LOGIT_RAG_STAGE1
        else:
            raise ValueError(f"Invalid prompt_build_method {prompt_build_method}")

        self.max_ctx_chars = int(max_ctx_chars)
        self.seed = seed
        self.cache = None
        self.generator = None

        self.logger.info("Initializing embedder...")
        self.embedder = build_embedder(embedder_cfg)

        self.logger.info("Loading index...")
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)

        self.logger.info("Checking dimension sanity...")
        dim = self.sanity_check_dimensions()

        self.logger.info("Loading docstore...")
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir, backend=docstore_backend)

        if cache_cfg is not None:
            self.cache = build_cache(cache_cfg, dim=dim, seed=self.seed)

        if not self.retrieve_only:
            self.logger.info("Initializing generator...")
            self.generator = build_generator(generator_cfg)

        self.logger.info(
            "RagPipeline initialized prompt=%s retrieve_only=%s top_k=%d embedder_device=%s generator_device=%s",
            self.prompt_build_method,
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

    # -------------------------
    # Helpers for retrieval / scoring
    # -------------------------
    def _retrieve_passages(self, question: str, k: int) -> tuple[list[Passage], list[str], dict[str, float], dict[str, Any]]:
        """
        Returns: passages, retrieved_doc_ids, timings, cache_info
        """
        passages: list[Passage] = []
        retrieved_doc_ids: list[str] = []
        timings: dict[str, float] = {}
        cache_info = {"cache_used": False, "cache_hits": 0, "cache_misses": 0}

        if k <= 0:
            timings.update({"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
            return passages, retrieved_doc_ids, timings, cache_info

        with timed(timings, "embed_s"):
            qvec = self.embedder.embed_queries([question])
            if hasattr(qvec, "detach"):
                qvec = qvec.detach().cpu().numpy()
            qvec = np.asarray(qvec, dtype=np.float32)

        cache_stats: dict[str, Any] | None = None
        with timed(timings, "ann_s"):
            if self.cache is not None:
                cache_info["cache_used"] = True
                distances, indices, cache_stats = self.cache.cached_search(qvec, k=k, backend_index=self.index)
            else:
                distances, indices = self.index.search(qvec, k)

        if cache_stats:
            cache_info["cache_hits"] = int(cache_stats.get("hits", 0))
            cache_info["cache_misses"] = int(cache_stats.get("misses", 0))

        with timed(timings, "docstore_s"):
            for rank, pid in enumerate(indices[0]):
                if pid < 0:
                    continue
                doc = self.docstore.get(str(pid))
                if not doc:
                    raise ValueError(
                        f"Docstore missing pid {pid} returned by index. This indicates data inconsistency between index and docstore."
                    )
                passages.append(
                    Passage(
                        pid=int(pid),
                        title=doc.get("title", ""),
                        text=doc.get("text", ""),
                        score=float(distances[0][rank]),
                    )
                )

        retrieved_doc_ids = [str(p.pid) for p in passages]
        return passages, retrieved_doc_ids, timings, cache_info

    def _mine_candidates(self, passages: list[Passage], max_candidates: int = 50) -> list[tuple[str, float]]:
        """
        Cheap candidate miner for Stage-1:
          - capitalized phrases (1-4 words)
          - numbers (incl. years)
          - basic date patterns
        Returns list of (candidate_string, prior_score) sorted descending.
        """
        # Weight earlier passages higher (rank prior)
        # Also weight by "closeness": if score is distance, smaller is better.
        # We don’t know your index metric, so we use rank-only by default.
        cand_scores: dict[str, float] = collections.defaultdict(float)

        cap_phrase = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
        # numbers / years
        num_pat = re.compile(r"\b\d{1,4}\b")
        year_pat = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
        # simple month date mentions
        month_pat = re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            re.IGNORECASE,
        )

        for rank, p in enumerate(passages):
            rank_w = 1.0 / (1.0 + rank)

            text = (p.title + "\n" + (p.text or ""))[:20000]  # avoid pathological long docs

            # Capitalized phrases (good for names/orgs/places)
            for m in cap_phrase.findall(text):
                c = m.strip()
                if len(c) < 3:
                    continue
                # filter common sentence starters that pollute candidates
                if c in {"The", "A", "An"}:
                    continue
                cand_scores[c] += 1.0 * rank_w

            # Numbers (good for years, counts)
            for m in num_pat.findall(text):
                cand_scores[m] += 0.5 * rank_w

            for m in year_pat.findall(text):
                cand_scores[m] += 1.0 * rank_w

            # If month appears, try to grab a nearby day/year pattern (very rough)
            if month_pat.search(text):
                # e.g., "January 12, 1999" / "Jan 12 1999"
                date_pat = re.compile(
                    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                    r"\s+\d{1,2}(?:,?\s+\d{4})?\b",
                    re.IGNORECASE,
                )
                for m in date_pat.findall(text):
                    cand_scores[m.strip()] += 1.0 * rank_w

        # Dedup with light normalization: collapse whitespace
        normalized_map: dict[str, str] = {}
        merged: dict[str, float] = collections.defaultdict(float)
        for c, s in cand_scores.items():
            key = " ".join(c.split())
            normalized_map[key] = c  # keep first surface form
            merged[key] += s

        items = sorted(((normalized_map[k], v) for k, v in merged.items()), key=lambda x: x[1], reverse=True)
        return items[:max_candidates]

    def _score_candidate_with_llm(self, question: str, candidate: str) -> float:
        """
        Score candidate as an answer using the generator (logprob / likelihood).
        Requires generator to expose one of:
          - score_chat(messages, completion) -> float (logprob)
          - score(prompt, completion) -> float (logprob)
        """
        if self.generator is None:
            raise RuntimeError("Generator is not initialized.")

        # A very simple “answer slot” prompt
        base_messages = [
            {"role": "user", "content": f"Question: {question}\nAnswer:"}
        ]
        completion = " " + candidate.strip()

        # Preferred: native chat scoring
        if hasattr(self.generator, "score_chat") and callable(getattr(self.generator, "score_chat")):
            return float(self.generator.score_chat(base_messages, completion))

        # Fallback: non-chat scoring
        if hasattr(self.generator, "score") and callable(getattr(self.generator, "score")):
            prompt = base_messages[0]["content"]
            return float(self.generator.score(prompt, completion))

        raise NotImplementedError(
            "Your generator wrapper does not expose score_chat(...) or score(...). "
            "Add a scoring method that returns log P(completion | prompt/messages)."
        )

    # ------------------------------------------------
    # Stage-1 "logit RAG": retrieve -> mine candidates -> LLM reranks -> pick best
    # ------------------------------------------------
    def run_logit_rag_stage1(
        self,
        question: str,
        *,
        max_candidates: int = 40,
        score_top_n: int = 20,
        length_normalize: bool = True,
        alpha_prior: float = 0.0,
    ) -> dict[str, Any]:
        """
        Stage 1:
          1) Retrieve top_k passages (NO prompt stuffing)
          2) Mine candidate answer strings from retrieved docs
          3) Score candidates with LLM log-likelihood
          4) Choose best candidate

        alpha_prior: if >0, mixes in mined prior (rank/freq) with model score:
          final = llm_score + alpha_prior * log(prior + eps)
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("question must be non-empty")
        if self.retrieve_only or self.generator is None:
            raise RuntimeError("Stage1 requires generator (retrieve_only=False).")
        if self.top_k <= 0:
            raise ValueError("Stage1 requires retrieval. Set top_k > 0 for run_logit_rag_stage1().")

        timings: dict[str, float] = {}
        cache_used = False
        cache_hits = 0
        cache_misses = 0

        # Build the scoring prompt ONCE (typed slot like Person:/Number:/Date:)
        scoring_messages = build_stage1_scoring_messages(question)

        # Retrieval
        with timed(timings, "retrieve_total_s"):
            passages, retrieved_doc_ids, rt, cache_info = self._retrieve_passages(question, k=self.top_k)
            
        timings.update(rt)
        cache_used = bool(cache_info.get("cache_used", False))
        cache_hits = int(cache_info.get("cache_hits", 0))
        cache_misses = int(cache_info.get("cache_misses", 0))

        # Candidate mining
        with timed(timings, "candidate_mine_s"):
            mined = self._mine_candidates(passages, max_candidates=max_candidates)


        if not mined:
            # fallback: just answer with plain LLM (no retrieval prompt)
            with timed(timings, "decode_s"):
                gen = self.generator.generate_chat(scoring_messages)
            return {
                "question": question,
                "mode": "logit_rag_stage1_fallback_llm",
                "messages": scoring_messages,
                "answer": gen.text,
                "raw_answer": gen.text,
                "retrieved_doc_ids": retrieved_doc_ids,
                "candidates": [],
                "best_candidate": "",
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "total_tokens": gen.total_tokens,
                "finish_reason": gen.metrics.get("finish_reason") or "",
                "timings_s": timings,
                "cache_used": cache_used,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }

       # Keep top-N by prior for scoring
        to_score = mined[: max(1, score_top_n)]

        # Normalize priors for optional mixing
        prior_vals = np.array([max(0.0, s) for _, s in to_score], dtype=np.float64)
        prior_sum = float(prior_vals.sum())
        if prior_sum > 0:
            prior_vals = prior_vals / prior_sum
        else:
            prior_vals = np.ones_like(prior_vals) / len(prior_vals)

        # LLM reranking
        scored: list[dict[str, Any]] = []
        with timed(timings, "candidate_score_s"):
            for i, (cand, _prior_unused) in enumerate(to_score):
                # KEY CHANGE: score candidate using the typed scoring prompt
                # This requires your generator to have score_chat().
                completion = " " + cand.strip()
                llm_score = float(self.generator.score_chat(scoring_messages, completion))

                # length norm
                if length_normalize:
                    denom = max(1, len(cand.split()))
                    llm_score = llm_score / denom

                # optional prior mixing
                if alpha_prior and alpha_prior > 0:
                    p = float(prior_vals[i])
                    llm_score = llm_score + float(alpha_prior) * math.log(p + 1e-12)

                scored.append(
                    {
                        "candidate": cand,
                        "prior": float(prior_vals[i]),
                        "llm_score": float(llm_score),
                    }
                )

        scored.sort(key=lambda x: x["llm_score"], reverse=True)
        best = scored[0]["candidate"] if scored else ""

        result = {
            "question": question,
            "mode": "logit_rag_stage1",
            "messages": scoring_messages,  # helpful for debugging
            "answer": best,
            "raw_answer": best,
            "retrieved_doc_ids": retrieved_doc_ids,
            "candidates": scored,
            "best_candidate": best,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reason": "",
            "timings_s": timings,
            "cache_used": cache_used,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }
        return result

    # ------------------------------------------------
    # Main entry point (used by gRPC)
    # ------------------------------------------------
    def run_prompt_rag(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must be non-empty")

        no_retrieval = (self.top_k == 0)
        passages: list[Passage] = []
        retrieved_doc_ids: list[str] = []
        timings: dict[str, float] = {}
        cache_used = False
        cache_hits = 0
        cache_misses = 0

        # Retrieval
        if no_retrieval:
            timings.update({"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
        else:
            passages, retrieved_doc_ids, rt, cache_info = self._retrieve_passages(question, k=self.top_k)
            timings.update(rt)
            cache_used = bool(cache_info.get("cache_used", False))
            cache_hits = int(cache_info.get("cache_hits", 0))
            cache_misses = int(cache_info.get("cache_misses", 0))

        # Early exit (retrieve-only)
        if self.retrieve_only or self.generator is None:
            timings.update({"prompt_s": 0.0, "decode_s": 0.0})
            return {
                "answer": "",
                "raw_answer": "",
                "retrieved_doc_ids": retrieved_doc_ids,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "timings_s": timings,
                "cache_used": cache_used,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }

        # Prompt construction
        with timed(timings, "prompt_s"):
            messages, _ = build_rag_messages(question, passages, self.prompt_build_method)

        # Generation
        with timed(timings, "decode_s"):
            gen = self.generator.generate_chat(messages)

        answer = gen.text
        prompt_tokens = gen.prompt_tokens
        completion_tokens = gen.completion_tokens
        total_tokens = gen.total_tokens

        timings["ttft_s"] = gen.metrics.get("ttft_s") or 0.0
        timings["prefill_tps"] = gen.metrics.get("prefill_tps") or 0.0
        timings["decode_tps"] = gen.metrics.get("decode_tps") or 0.0
        finish_reason = gen.metrics.get("finish_reason") or ""

        result = {
            "question": question,
            "messages": messages,
            "answer": answer,
            "raw_answer": answer,
            "retrieved_doc_ids": retrieved_doc_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "timings_s": timings,
            "cache_used": cache_used,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }

        if self.always_log_results:
            self.log_result(result)

        return result

    def log_result(self, result: dict[str, Any], log_path: Optional[str] = None):
        append_csv_row(log_path, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": result.get("question", ""),
            "top_k": self.top_k,
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "total_tokens": result.get("total_tokens", 0),
            "ttft_s": result.get("timings_s", {}).get("ttft_s", 0.0),
            "total_s": result.get("timings_s", {}).get("total_s", 0.0),
            "prefill_tps": result.get("timings_s", {}).get("prefill_tps", 0.0),
            "decode_tps": result.get("timings_s", {}).get("decode_tps", 0.0),
            "embed_s": result.get("timings_s", {}).get("embed_s", 0.0),
            "ann_s": result.get("timings_s", {}).get("ann_s", 0.0),
            "docstore_s": result.get("timings_s", {}).get("docstore_s", 0.0),
            "prompt_build_s": result.get("timings_s", {}).get("prompt_s", 0.0),
            "finish_reason": result.get("finish_reason", ""),
            "cache_used": result.get("cache_used", 0),
            "cache_hits": result.get("cache_hits", 0),
            "cache_misses": result.get("cache_misses", 0),
        })
