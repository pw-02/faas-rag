from __future__ import annotations

import logging
import time
import re
import math
import collections
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple

import numpy as np
from datetime import datetime, timezone
from contextlib import contextmanager

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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
from faasrag.core.prompts import (
    PromptBuildMethodType,
    build_rag_messages,
    build_scoring_messages,
    
)
from faasrag.core.utils import (
    append_csv_row, dedupe_overlapping_phrases,
      top_biased_tokens_pairs)



@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0

# -------------------------
# Result types
# -------------------------
@dataclass
class RetrievalResult:
    passages: List[Passage] = field(default_factory=list)
    doc_ids: List[str] = field(default_factory=list)
    timings_s: Dict[str, float] = field(default_factory=dict)
    cache_used: bool = False
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class RagRunResult:
    mode: str
    question: str
    answer: str
    raw_answer: str
    messages: List[dict] = field(default_factory=list)
    retrieved_doc_ids: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    timings_s: Dict[str, float] = field(default_factory=dict)
    # For extra diagnostics without breaking schema
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "mode": self.mode,
            "question": self.question,
            "messages": self.messages,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
            "finish_reason": self.finish_reason,
            "timings_s": self.timings_s,
        }
        d.update(self.extra or {})
        return d


# -------------------------
# Pipeline
# -------------------------
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
        self.max_ctx_chars = int(max_ctx_chars)
        self.seed = seed

        pbm = (prompt_build_method or "").upper().strip()
        if pbm == "QA_STRICT":
            self.prompt_build_method = PromptBuildMethodType.QA_STRICT
        elif pbm == "QA_OPEN":
            self.prompt_build_method = PromptBuildMethodType.QA_OPEN
        elif pbm == "FEW_SHOT":
            self.prompt_build_method = PromptBuildMethodType.FEW_SHOT
        elif pbm == "LLM_ONLY":
            self.prompt_build_method = PromptBuildMethodType.LLM_ONLY
        elif pbm == "LOGIT_RAG_STAGE1":
            self.prompt_build_method = PromptBuildMethodType.LOGIT_RAG_STAGE1
        elif pbm == "LOGIT_RAG":
            self.prompt_build_method = PromptBuildMethodType.LOGIT_RAG
        else:
            raise ValueError(f"Invalid prompt_build_method {prompt_build_method}")
        
        self.logger.info("Initializing RagPipeline with prompt_build_method=%s", self.prompt_build_method)

        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.top_k == 0 or self.retrieve_only:
            self.logger.warning("No retrieval will be performed (top_k=0 or retrieve_only=True).")

        # Embedder
        self.logger.info("Initializing embedder...")
        self.embedder = build_embedder(embedder_cfg)

        # Index
        self.logger.info("Loading index...")
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)

        # Sanity check dims
        self.logger.info("Checking dimension sanity...")
        dim = self._sanity_check_dimensions()

        # Docstore
        self.logger.info("Loading docstore...")
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir, backend=docstore_backend)

        # Cache
        self.cache = build_cache(cache_cfg, dim=dim, seed=self.seed) if cache_cfg is not None else None

        # Generator
        self.generator = None
        if not self.retrieve_only:
            self.logger.info("Initializing generator...")
            self.generator = build_generator(generator_cfg)

        # Reader config (lazy loaded)
        self._reader_initialized = False
        self.reader_name = getattr(generator_cfg, "reader_name", None) or "deepset/roberta-base-squad2"
        self.reader_device = getattr(generator_cfg, "reader_device", None) or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reader_tokenizer = None
        self.reader_model = None

        self.logger.info(
            "RagPipeline initialized prompt=%s retrieve_only=%s top_k=%d embedder_device=%s generator_device=%s",
            self.prompt_build_method,
            self.retrieve_only,
            self.top_k,
            getattr(self.embedder, "device", "unknown"),
            getattr(self.generator, "device", "unknown"),
        )

    # -------------------------
    # Setup helpers
    # -------------------------
    def _sanity_check_dimensions(self) -> int:
        test = self.embedder.embed_queries(["dim check"])
        embed_dim = int(test.shape[1])

        index_dim = getattr(self.index, "d", None)
        if index_dim is None:
            raise ValueError(f"Index object {type(self.index)} has no attribute `.d`")
        if embed_dim != int(index_dim):
            raise ValueError(f"Embed dim {embed_dim} != index dim {index_dim}")

        self.logger.info("Embedder dim %d matches index dim %d", embed_dim, int(index_dim))
        return embed_dim

    def _ensure_reader(self) -> None:
        if self._reader_initialized:
            return
        self.logger.info("Lazy-loading reader QA model for mining: %s", self.reader_name)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(self.reader_name, use_fast=True)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(self.reader_name)
        self.reader_model.eval()
        self.reader_model.to(self.reader_device)
        self.logger.info("Reader moved to %s", self.reader_device)
        self._reader_initialized = True

    # -------------------------
    # Retrieval
    # -------------------------
    def _retrieve(self, question: str, k: int) -> RetrievalResult:
        rr = RetrievalResult()
        if k <= 0:
            rr.timings_s.update({"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
            return rr

        with timed(rr.timings_s, "embed_s"):
            qvec = self.embedder.embed_queries([question])
            if hasattr(qvec, "detach"):
                qvec = qvec.detach().cpu().numpy()
            qvec = np.asarray(qvec, dtype=np.float32)

        cache_stats: dict[str, Any] | None = None
        with timed(rr.timings_s, "ann_s"):
            if self.cache is not None:
                rr.cache_used = True
                distances, indices, cache_stats = self.cache.cached_search(
                    qvec, k=k, backend_index=self.index
                )
            else:
                distances, indices = self.index.search(qvec, k)

        if cache_stats:
            rr.cache_hits = int(cache_stats.get("hits", 0))
            rr.cache_misses = int(cache_stats.get("misses", 0))

        with timed(rr.timings_s, "docstore_s"):
            for rank, pid in enumerate(indices[0]):
                if pid < 0:
                    continue
                doc = self.docstore.get(str(pid))
                if not doc:
                    raise ValueError(f"Docstore missing pid {pid} returned by index.")
                rr.passages.append(
                    Passage(
                        pid=int(pid),
                        title=doc.get("title", ""),
                        text=doc.get("text", ""),
                        score=float(distances[0][rank]),
                    )
                )

        rr.doc_ids = [str(p.pid) for p in rr.passages]
        return rr
    
    def _candidates_to_token_bias(
        self,
        scored_phrases: list[tuple[str, float]],
        *,
        max_token_logit_bias: float | None = None,
        phrase_softmax_temperature: float = 1.0,
        drop_junk_tokens: bool = True,
        dedupe_overlaps: bool = True,
        split_weight_across_tokens: bool = False,  # keep False for your “don’t divide” fix
    ) -> dict[int, float]:
        """
        Convert mined candidate phrases into token_id -> logit_bias map.

        Steps:
        1) (Optional) dedupe overlapping phrases (drop "Bruce" if "Bruce Springsteen" exists).
        2) Softmax weights over phrase scores: w_i = softmax(score_i / T)
            - Lower T (<1) => sharper (top phrase dominates)
            - Higher T (>1) => flatter (weights more uniform)
        3) Convert each phrase to token IDs and add weight to each token.
            - If split_weight_across_tokens=True: per_tok = w / len(tokens)
            - Else: per_tok = w  (stronger for multi-token phrases)
        4) (Optional) drop junk tokens and cap per-token bias.

        When you increase temperature: 
            -The top-scoring phrase becomes less dominant.
            - Lower-scoring phrases get relatively more weight
            - Bias is distributed across more candidates

        NOTE: This is token-level steering; very large bias can cause repetition loops.
        """
        if not scored_phrases:
            return {}
        if self.generator is None or not hasattr(self.generator, "tokenizer"):
            raise RuntimeError("Need HF generator exposing .tokenizer for stage2 biasing.")

        tokenizer = self.generator.tokenizer
        special_ids = set(getattr(tokenizer, "all_special_ids", []))

        items = scored_phrases
        if dedupe_overlaps:
            items = dedupe_overlapping_phrases(items)

        if not items:
            return {}

        # --- softmax over phrase scores ---
        scores = np.array([float(score) for _phrase, score in items], dtype=np.float64)
        T = max(1e-9, float(phrase_softmax_temperature))
        logits = scores / T
        logits = logits - logits.max()
        w = np.exp(logits)
        w = w / (w.sum() + 1e-9)

        def is_junk_token_id(tid: int) -> bool:
            if tid in special_ids:
                return True
            if not drop_junk_tokens:
                return False
            s = tokenizer.decode([tid]).strip()
            if s == "":
                return True
            if re.fullmatch(r"[^\w]+", s):
                return True
            if len(s) == 1 and not s.isalnum():
                return True
            return False

        token_bias: Dict[int, float] = {}

        for (phrase, _score), phrase_w in zip(items, w.tolist()):
            phrase = (phrase or "").strip()
            if not phrase:
                continue

            token_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue

            # --- key change: don't divide by token count (default) ---
            if split_weight_across_tokens:
                per_tok = float(phrase_w) / max(1, len(token_ids))
            else:
                per_tok = float(phrase_w)

            for tid in token_ids:
                tid = int(tid)
                if is_junk_token_id(tid):
                    continue
                token_bias[tid] = token_bias.get(tid, 0.0) + per_tok

        # cap
        if max_token_logit_bias is not None and max_token_logit_bias > 0:
            cap = float(max_token_logit_bias)
            for tid in list(token_bias.keys()):
                if token_bias[tid] > cap:
                    token_bias[tid] = cap

        return token_bias
  
    # -------------------------
    # Candidate mining (QA reader -> atomic candidate strings)
    # -------------------------
    def _mine_candidates_from_passages(
        self,
        question: str,
        passages: list[Passage],
        *,
        max_mined_candidates: int = 50,
        per_passage_nbest: int = 8,
        max_answer_chars: int = 80,
        max_seq_len: int = 384,
        doc_stride: int = 128,
        max_span_tokens: int = 8,
        length_penalty: float = 0.35,
    ) -> list[tuple[str, float]]:
        """
        PURPOSE
        -------
        Extract a *small set of plausible short answers* from retrieved passages, using an
        extractive QA model (reader). Returns candidates with a score that combines:
        - reader confidence (start+end logits, length-penalized)
        - passage relevance weight (based on retriever score + rank)

        OUTPUT
        ------
        List of (candidate_text, score), sorted descending. Candidates are "atomic"
        strings (names / years / numbers / short noun phrases), not long spans/sentences.

        This output is used by:
        - Stage 1: rerank candidates by LLM log-likelihood (selection)
        - Stage 2: convert candidates into token logit bias (generation prior)

        What happens (in order)
            1. For each retrieved passage (and each sliding window of that passage)
                you run the reader QA model to get start_logits and end_logits.
            2. You form lots of candidate spans by pairing high-scoring start positions with high-scoring end positions, 
            3. You turn those spans into text using offset_mapping to map token indices → character offsets → substring of the passage.
            4. You refine/clean them (your regex / heuristics)and optionally extract “atomic” subspans (years, numbers+units, name-like chunks)
            5. You aggregate scores across passages/windows into cand_scores.If the same candidate appears 
                multiple times (or in strong passages), it rises.
            6. You deduplicate/merge near-duplicates
            7. Finally you sort candidates and return the top max_mined_candidates as [(candidate_string, aggregated_score), ...].
        """
        self._ensure_reader()
        assert self.reader_model is not None
        assert self.reader_tokenizer is not None

        # -----------------------------
        # 0) Basic input hygiene
        # -----------------------------
        question = (question or "").strip()
        if not question:
            return []

        # -----------------------------
        # 1) Very cheap question typing
        #    (used to filter/refine extracted spans)
        # -----------------------------
        qlow = question.lower().strip()
        is_who = qlow.startswith("who") or " who " in f" {qlow} "
        is_when = qlow.startswith("when") or " what year" in qlow or " what date" in qlow
        is_where = qlow.startswith("where")
        is_how_many = qlow.startswith("how many") or qlow.startswith("how much")

        # -----------------------------
        # 2) Compute per-passage weights
        #
        # Idea: candidates from higher-ranked / more-similar passages should count more.
        # - sim_w: softmax over retriever similarity scores
        # - rank_w: 1/(1+i) to reward early passages
        # - passage_w: blended weight used later during aggregation
        # -----------------------------
        sims = [float(getattr(p, "score", 0.0) or 0.0) for p in passages]
        if sims:
            m = max(sims)
            exps = [math.exp(s - m) for s in sims]
            Z = sum(exps) or 1.0
            sim_w = [e / Z for e in exps]  # sums to 1
        else:
            sim_w = [1.0 / max(1, len(passages))] * len(passages)

        # rank_w = [1.0 / (1.0 + i) for i in range(len(passages))]
        rank_w = [1.0 / math.sqrt(1.0 + i) for i in range(len(passages))]  # softer decay
        
        #rank_w = [1.0] * len(passages) #no decay based on rank at all. This is to test whether the similarity weighting alone is sufficient to differentiate passage importance, without the need for an additional rank-based decay.

        # Blend: rank weighting plus similarity weighting
        passage_w: list[float] = []
        for i in range(len(passages)):
            # (0.2 + 0.8 * ...) keeps weights from collapsing too hard
            w = rank_w[i] * (0.2 + 0.8 * sim_w[i] * len(passages))
            passage_w.append(float(w))

        # -----------------------------
        # 3) Regex + span filters to enforce "atomic" outputs
        #
        # Key principle: your QA reader often returns fragments.
        # We aggressively normalize, reject, and "refine" them into:
        #   - names (for who/where)
        #   - years/dates/numbers (for when/how many)
        # -----------------------------
        ws_re = re.compile(r"\s+")
        bad_punct_re = re.compile(r"[\.!\?;]")    # sentence-ish punctuation => likely not an atomic answer
        many_commas_re = re.compile(r",.*,")      # multiple commas => likely list/phrase, not atomic

        year_pat = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
        number_pat = re.compile(r"\b\d+(\.\d+)?\b")
        unit_pat = re.compile(
            r"\b(pages?|years?|months?|days?|km|kilometers?|miles?|meters?|feet|ft|inches?|%|dollars?|usd|euros?|pounds?)\b",
            re.IGNORECASE,
        )

        # NOTE: name_pat.findall() returns ONLY the *capturing group* unless you use non-capturing groups.
        # In your current pattern, ([A-Z][a-z]+|[A-Z]\.) is a capturing group, so findall()
        # returns just the FIRST token (e.g., "Charles") not "Charles Cornwallis".
        #
        # If you want full spans, change to a non-capturing group:
        #   r"\b(?:[A-Z][a-z]+|[A-Z]\.)(?:\s+(?:...))*\b"
        #

        # name_pat = re.compile(
        #     r"\b([A-Z][a-z]+|[A-Z]\.)"
        #     r"(?:\s+(?:[A-Z][a-z]+|[A-Z]\.|de|da|del|van|von|al|bin|ibn|la|le|of))*\b"
        # )

        #change to non-capturing group to get full name spans instead of just first token
        name_pat = re.compile(
            r"\b(?:[A-Z][a-z]+|[A-Z]\.)"
            r"(?:\s+(?:[A-Z][a-z]+|[A-Z]\.|de|da|del|van|von|al|bin|ibn|la|le|of))*\b")


        def clean_span(s: str) -> str:
            """Collapse whitespace + strip edge punctuation/quotes so spans compare/dedup cleanly."""
            s = (s or "").strip()
            s = ws_re.sub(" ", s)
            return s.strip(" \t\r\n\"'`.,;:()[]{}")

        def too_long_or_short(s: str) -> bool:
            """Reject empty, 1-char, too-long, or too-many-words spans."""
            if not s or len(s) < 2 or len(s) > max_answer_chars:
                return True
            toks = s.split()
            return (len(toks) == 0) or (len(toks) > max_span_tokens)

        def is_junk(s: str) -> bool:
            """
            Reject spans that are likely useless as answers:
            - stopword-only
            - contain sentence punctuation
            - look like long comma-separated fragments
            - mostly non-alphanumeric
            """
            sl = s.lower()
            if sl in {"the", "a", "an", "it", "they", "he", "she", "this", "that", "these", "those", "sir", "madam"}:
                return True
            if bad_punct_re.search(s):
                return True
            if many_commas_re.search(s):
                return True
            alnum = sum(ch.isalnum() for ch in s)
            return alnum < max(2, int(0.4 * len(s)))

        def type_mismatch(s: str) -> bool:
            """
            Light gating: if it's a 'who' question, candidate should look name-like.
            If it's a 'when/how many', must include digits, etc.
            """
            if is_who:
                return not bool(re.search(r"\b[A-Z][a-z]+\b", s))
            if is_when:
                return not (bool(year_pat.search(s)) or bool(number_pat.search(s)))
            if is_how_many:
                return not bool(number_pat.search(s))
            if is_where:
                return not bool(re.search(r"\b[A-Z][a-z]+\b", s))
            return False

        def refine_atomic(s: str) -> list[str]:
            """
            Given a raw reader span, produce *atomic subspans*:
            - number(+unit) chunks, years (when/how many)
            - name-like chunks (who/where)
            - plus the cleaned original if it already looks atomic

            This is the key step that turns messy spans into usable candidate strings.
            """
            out: list[str] = []
            s2 = clean_span(s)
            if not s2 or is_junk(s2):
                return out

            # Pull numeric chunks (optionally with a unit)
            if is_how_many or is_when:
                for m in re.finditer(
                    r"\b\d{1,6}(?:\.\d+)?(?:\s+" + unit_pat.pattern[2:-2] + r")?\b",
                    s2,
                    re.IGNORECASE,
                ):
                    out.append(clean_span(m.group(0)))

            # Pull standalone years
            if is_when:
                for m in year_pat.findall(s2):
                    out.append(clean_span(m))

            # Pull name-like phrases (see NOTE above about findall capturing group!)
            if is_who or is_where:
                for m in name_pat.findall(s2):
                    c = clean_span(m)
                    if c and len(c.split()) <= max_span_tokens:
                        out.append(c)

            # If the whole span is already short+clean, keep it too.
            if (not too_long_or_short(s2)) and (not is_junk(s2)):
                out.append(s2)

            # Deduplicate in order (case/whitespace insensitive)
            seen = set()
            uniq: list[str] = []
            for x in out:
                k = " ".join(x.lower().split())
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(x)
            return uniq

        # -----------------------------
        # 4) Run extractive QA over each passage
        #
        # Reader returns start/end logits for each window (sliding over long passages).
        # We pick top start indices and end indices, form spans, length-penalize,
        # and keep per_passage_nbest spans per passage window.
        # -----------------------------
        cand_scores: dict[str, float] = collections.defaultdict(float)
        device = next(self.reader_model.parameters()).device

        with torch.no_grad():
            for i, p in enumerate(passages):
                # reader context is (title + text), truncated to keep runtime bounded
                ctx = ((p.title or "") + "\n" + (p.text or ""))[:20000]
                if not ctx.strip():
                    continue

                # Tokenize (question, context) with sliding windows
                enc = self.reader_tokenizer(
                    question,
                    ctx,
                    # truncation="only_second",
                    truncation=True,
                    max_length=max_seq_len,
                    stride=doc_stride,
                    return_overflowing_tokens=True,   # multiple windows for long contexts
                    return_offsets_mapping=True,      # map token positions -> character spans
                    padding=True,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                offset_mapping = enc["offset_mapping"]
                seq_ids_fn = getattr(enc, "sequence_ids", None)  # tells which tokens are question vs context

                out = self.reader_model(input_ids=input_ids, attention_mask=attn)
                start_logits = out.start_logits
                end_logits = out.end_logits

                # Iterate each window produced by the sliding tokenizer
                for widx in range(start_logits.size(0)):
                    # Move logits to CPU for topk ops (can keep on GPU if you want speed)
                    s_logits = start_logits[widx].detach().cpu()
                    e_logits = end_logits[widx].detach().cpu()

                    # Identify context tokens (sequence_id == 1 typically indicates context)
                    if callable(seq_ids_fn):
                        seq_ids = seq_ids_fn(widx)
                    else:
                        seq_ids = [0] * len(offset_mapping[widx])

                    # valid token positions = context tokens with a non-empty char span
                    valid: list[int] = []
                    for tidx, off in enumerate(offset_mapping[widx]):
                        if off is None:
                            continue
                        if seq_ids[tidx] == 1 and off[1] > off[0]:
                            valid.append(tidx)

                    # fallback: if sequence_ids unavailable, accept any token with offset span
                    if not valid:
                        valid = [
                            tidx for tidx, off in enumerate(offset_mapping[widx])
                            if off is not None and off[1] > off[0]
                        ]
                    if not valid:
                        continue

                    # Choose top start and end positions (broader than per_passage_nbest, then cross-product)
                    k = max(10, per_passage_nbest * 6)
                    vs = s_logits[valid]
                    ve = e_logits[valid]
                    top_s = torch.topk(vs, k=min(k, vs.numel()))
                    top_e = torch.topk(ve, k=min(k, ve.numel()))

                    top_s_idx = [valid[int(ix)] for ix in top_s.indices.tolist()]
                    top_e_idx = [valid[int(ix)] for ix in top_e.indices.tolist()]

                    # Form candidate spans from start/end pairs (bounded by max_span_tokens)
                    spans: list[tuple[int, int, float]] = []
                    for si in top_s_idx:
                        for ei in top_e_idx:
                            if ei < si:
                                continue
                            span_len = (ei - si) + 1
                            if span_len > max_span_tokens:
                                continue

                            # Reader confidence ~ start_logit + end_logit
                            raw_score = float(s_logits[si] + e_logits[ei])

                            # Penalize longer spans so single tokens / short names win
                            adj_score = raw_score - (length_penalty * span_len)
                            spans.append((si, ei, adj_score))

                    # Keep only best few spans per window
                    spans.sort(key=lambda x: x[2], reverse=True)
                    spans = spans[:per_passage_nbest]

                    # Convert token spans -> character spans -> raw text -> refined atomic candidates
                    for si, ei, span_score in spans:
                        off_s = offset_mapping[widx][si]
                        off_e = offset_mapping[widx][ei]
                        if off_s is None or off_e is None:
                            continue

                        start_char = int(off_s[0])
                        end_char = int(off_e[1])
                        if end_char <= start_char:
                            continue

                        raw = ctx[start_char:end_char]

                        # refine_atomic() is where we split messy spans into atomic pieces
                        refined = refine_atomic(clean_span(raw))
                        if not refined:
                            continue

                        # Aggregate candidate scores:
                        # passage_w[i] emphasizes good passages
                        # (1 + span_score) keeps sign positive-ish and preserves ranking
                        for c in refined:
                            if too_long_or_short(c) or is_junk(c) or type_mismatch(c):
                                continue
                            cand_scores[c] += passage_w[i] * (1.0 + span_score)

        # -----------------------------
        # 5) Merge + deduplicate by normalized form
        #    (e.g. "Charles Cornwallis" vs "charles cornwallis")
        # -----------------------------
        def norm_key(s: str) -> str:
            s = s.lower().strip()
            s = ws_re.sub(" ", s)
            return s.strip(" \t\r\n\"'`.,;:()[]{}")

        merged: dict[str, float] = collections.defaultdict(float)
        surface: dict[str, str] = {}

        for c, s in cand_scores.items():
            k = norm_key(c)
            if not k:
                continue

            # Keep a "best" surface form for display (prefer longer name-like forms)
            if k not in surface:
                surface[k] = c
            else:
                if len(c) > len(surface[k]) and len(c.split()) <= max_span_tokens:
                    surface[k] = c

            merged[k] += float(s)

        # Sort by merged score and cap output size
        items = sorted(((surface[k], v) for k, v in merged.items()), key=lambda x: x[1], reverse=True)
        return items[:max_mined_candidates]


    # -------------------------
    # Public runs
    # -------------------------
    def run_prompt_rag(self, question: str) -> dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")
        start = time.perf_counter()
        timings: dict[str, float] = {}
        rr = self._retrieve(q, self.top_k) if self.top_k > 0 else RetrievalResult(timings_s={"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
        timings.update(rr.timings_s)

        if self.retrieve_only or self.generator is None:
            res = RagRunResult(
                mode="prompt_rag_retrieve_only",
                question=q,
                answer="",
                raw_answer="",
                messages=[{"role": "user", "content": q}],
                retrieved_doc_ids=rr.doc_ids,
                timings_s=timings,
                extra={"cache_used": rr.cache_used, "cache_hits": rr.cache_hits, "cache_misses": rr.cache_misses},
            )
            return res.to_dict()

        with timed(timings, "prompt_s"):
            messages, _ = build_rag_messages(q, rr.passages, self.prompt_build_method)

        with timed(timings, "decode_s"):
            gen = self.generator.generate_chat(messages)

        timings["ttft_s"] = gen.metrics.get("ttft_s") or 0.0
        timings["prefill_tps"] = gen.metrics.get("prefill_tps") or 0.0
        timings["decode_tps"] = gen.metrics.get("decode_tps") or 0.0
        timings["total_s"] = time.perf_counter() - start

        res = RagRunResult(
            mode="prompt_rag",
            question=q,
            answer=gen.text,
            raw_answer=gen.text,
            messages=messages,
            retrieved_doc_ids=rr.doc_ids,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.total_tokens,
            finish_reason=gen.metrics.get("finish_reason") or "",
            timings_s=timings,
            extra={"cache_used": rr.cache_used, "cache_hits": rr.cache_hits, "cache_misses": rr.cache_misses},
        )
        if self.always_log_results:
            self.log_result(res.to_dict())
        

        return res.to_dict()
    

    def run_logit_rag_stage1(
        self,
        question: str,
        *,
        top_candidates: int = 40,
        score_top_n: int = 20,
        length_normalize: bool = True,
        alpha_prior: float = 0.0,
    ) -> dict[str, Any]:
        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")
        if self.retrieve_only or self.generator is None:
            raise RuntimeError("Stage1 requires generator (retrieve_only=False).")
        if self.top_k <= 0:
            raise ValueError("Stage1 requires retrieval (top_k > 0).")
        
        start = time.perf_counter()

        timings: dict[str, float] = {}
        # token accounting for stage-1 scoring passes
        score_prompt_tok_sum = 0
        score_completion_tok_sum = 0
        score_total_tok_sum = 0

        scoring_messages = build_scoring_messages(q)

        with timed(timings, "retrieve_total_s"):
            rr = self._retrieve(q, self.top_k)
        timings.update(rr.timings_s)

        with timed(timings, "candidate_mine_s"):
            mined = self._mine_candidates_from_passages(q, rr.passages, top_candidates=top_candidates)

        if not mined:
            with timed(timings, "decode_s"):
                gen = self.generator.generate_chat(scoring_messages)
            res = RagRunResult(
                mode="logit_rag_stage1_fallback_llm",
                question=q,
                answer=gen.text,
                raw_answer=gen.text,
                messages=scoring_messages,
                retrieved_doc_ids=rr.doc_ids,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                total_tokens=gen.total_tokens,
                finish_reason=gen.metrics.get("finish_reason") or "",
                timings_s=timings,
                extra={"cache_used": rr.cache_used, "cache_hits": rr.cache_hits, "cache_misses": rr.cache_misses, "candidates": [], "best_candidate": ""},
            )
            return res.to_dict()

        to_score = mined[: max(1, score_top_n)]
        prior_vals = np.array([max(0.0, s) for _, s in to_score], dtype=np.float64)
        prior_sum = float(prior_vals.sum())
        prior_vals = (prior_vals / prior_sum) if prior_sum > 0 else (np.ones_like(prior_vals) / len(prior_vals))

        scored: list[dict[str, Any]] = []
        
        with timed(timings, "candidate_score_s"):
            for i, (cand, _prior_unused) in enumerate(to_score):
                completion = " " + cand.strip()
                llm_score, p_tok, c_tok, t_tok = self.generator.score_chat(
                    scoring_messages,
                    completion,
                    length_normalize=False,  # keep your own normalization logic below
                )

                # Stage1 is not “generation tokens,” it’s scoring compute. Counting tokens this way makes comparisons honest:
                score_prompt_tok_sum += int(p_tok)
                score_completion_tok_sum += int(c_tok)
                score_total_tok_sum += int(t_tok)
                llm_score = float(llm_score)

                if length_normalize:
                    cand_ids = self.generator.tokenizer(" " + cand.strip(), add_special_tokens=False)["input_ids"]
                    denom = max(1, len(cand_ids))
                    llm_score = llm_score / denom

                if alpha_prior and alpha_prior > 0:
                    p = float(prior_vals[i])
                    llm_score = llm_score + float(alpha_prior) * math.log(p + 1e-12)

                scored.append({"candidate": cand, "prior": float(prior_vals[i]), "llm_score": float(llm_score)})

        scored.sort(key=lambda x: x["llm_score"], reverse=True)
        best = scored[0]["candidate"] if scored else ""
        timings["total_s"] = time.perf_counter() - start

        res = RagRunResult(
            mode="logit_rag_stage1",
            question=q,
            answer=best,
            raw_answer=best,
            messages=scoring_messages,
            retrieved_doc_ids=rr.doc_ids,
            prompt_tokens=int(score_prompt_tok_sum),
            completion_tokens=int(score_completion_tok_sum),
            total_tokens=int(score_total_tok_sum),
            timings_s=timings,
            extra={
                "cache_used": rr.cache_used,
                "cache_hits": rr.cache_hits,
                "cache_misses": rr.cache_misses,
                "candidates": scored,
                "best_candidate": best,
            },
        )

        return res.to_dict()
            
    def run_logit_rag(
        self,
        question: str,
        max_mined_candidates: int = 40,
        logit_bias_strength: float = 0.8,
        max_token_logit_bias: float = 2.0,
        phrase_softmax_temperature: float = 1.0,
        clamp_first_line: bool = True,
        hybrid_prompt: bool = False,
        max_bias_steps: Optional[int] = None,
        bias_top_n: Optional[int] = None,
        dedupe_overlaps: bool = True,

    ) -> dict[str, Any]:

        q = (question or "").strip()
        if not q:
            raise ValueError("question must be non-empty")
        if self.retrieve_only or self.generator is None:
            raise RuntimeError("Stage2 requires generator (retrieve_only=False).")
        if self.top_k <= 0:
            raise ValueError("Stage2 requires retrieval (top_k > 0).")
        
        if bias_top_n is not None and bias_top_n <= 0:
            raise ValueError("bias_top_n must be positive if specified.")
        if bias_top_n is None:
            bias_top_n = max_mined_candidates

        timings: dict[str, float] = {}
        start = time.perf_counter()

        with timed(timings, "retrieve_total_s"):
            retrieval_result = self._retrieve(q, self.top_k)
        timings.update(retrieval_result.timings_s)

        with timed(timings, "candidate_mine_s"):
            mined_candidates = self._mine_candidates_from_passages(
                q,
                retrieval_result.passages,
                max_mined_candidates=max_mined_candidates,
            )

        if dedupe_overlaps:
            mined_candidates = dedupe_overlapping_phrases(mined_candidates)
        
        with timed(timings, "token_bias_build_s"):
            ranked = sorted(mined_candidates, key=lambda x: float(x[1]), reverse=True)
            top_phrases = ranked[:bias_top_n] if bias_top_n and bias_top_n > 0 else []
            logit_bias = self._candidates_to_token_bias(
                scored_phrases=top_phrases,
                max_token_logit_bias=max_token_logit_bias,
                phrase_softmax_temperature=phrase_softmax_temperature,
                drop_junk_tokens=True,
                dedupe_overlaps=dedupe_overlaps,
                split_weight_across_tokens=False,
            )

        with timed(timings, "prompt_s"):
            # TODO: implement hybrid_prompt if you want different message construction
            messages, _ = build_rag_messages(q, retrieval_result.passages, self.prompt_build_method)

        with timed(timings, "decode_s"):
            gen = self.generator.generate_chat_with_logit_bias(
                messages=messages,
                bias=logit_bias,
                logit_bias_strength=logit_bias_strength,
                max_bias_steps=max_bias_steps,
                clamp_first_line=clamp_first_line,
            )

        timings["total_s"] = time.perf_counter() - start

        # mined_phrases = [p for p, _ in mined_candidates]
        top_bias = top_biased_tokens_pairs(self.generator.tokenizer, logit_bias, k=20)

        res = RagRunResult(
            mode="logit_rag",
            question=q,
            answer=gen.text,
            raw_answer=gen.text,
            messages=messages,
            retrieved_doc_ids=retrieval_result.doc_ids,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.total_tokens,
            finish_reason=gen.metrics.get("finish_reason") or "",
            timings_s=timings,
            extra={
                "cache_used": retrieval_result.cache_used,
                "cache_hits": retrieval_result.cache_hits,
                "cache_misses": retrieval_result.cache_misses,
                "mined_candidates": mined_candidates,   # phrase+score
                "num_biased_token_ids": len(logit_bias),
                "top_biased_tokens": top_bias,   # list of (token_id, bias_value)
                "logit_bias_strength": float(logit_bias_strength),
            },
        )
        return res.to_dict()


    def log_result(self, result: dict[str, Any], log_path: Optional[str] = None):
        append_csv_row(
            log_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": result.get("mode", ""),
                "question": result.get("question", ""),
                "top_k": self.top_k,
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "ttft_s": (result.get("timings_s") or {}).get("ttft_s", 0.0),
                "finish_reason": result.get("finish_reason", ""),
                "cache_used": int(bool(result.get("cache_used", False))),
                "cache_hits": int(result.get("cache_hits", 0) or 0),
                "cache_misses": int(result.get("cache_misses", 0) or 0),
            },
        )
