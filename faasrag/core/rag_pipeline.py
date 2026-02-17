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

# Put these imports at the top of your rag_pipeline.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0

def _candidates_to_token_bias(
    self,
    candidates: list[tuple[str, float]],
    *,
    max_phrases: int = 40,
    per_token_cap: float = 2.0,
    phrase_score_temperature: float = 1.0,
    drop_junk_tokens: bool = True,
) -> dict[int, float]:
    """
    Turn mined candidate strings into sparse token bias dict.

    candidates: list[(text, score)] sorted desc (from _mine_candidates_qa1)
    Returns: {token_id: bias_value} where bias_value is additive to logits (before alpha scaling).

    Design:
    - take top max_phrases candidate strings
    - tokenize each string with GENERATOR tokenizer (not reader tokenizer)
    - distribute the phrase score across its tokens (so multi-token names don't overpower)
    - accumulate per token
    - optionally filter out whitespace/punct tokens

    This is Option B: entity/keyword mining -> token IDs -> logit bias
    """
    if not candidates:
        return {}

    if not hasattr(self.generator, "tokenizer"):
        raise RuntimeError("Generator tokenizer not found; need HF generator exposing .tokenizer")

    tok = self.generator.tokenizer
    special_ids = set(getattr(tok, "all_special_ids", []))

    # Use only top phrases
    items = candidates[:max_phrases]

    # Normalize phrase scores so they are stable across queries
    scores = np.array([float(s) for _, s in items], dtype=np.float64)
    # Softmax-ish normalization (temperature)
    scores = scores / max(1e-9, float(phrase_score_temperature))
    scores = scores - scores.max()
    w = np.exp(scores)
    w = w / (w.sum() + 1e-9)  # sums to 1

    def is_junk_token_id(tid: int) -> bool:
        if tid in special_ids:
            return True
        if not drop_junk_tokens:
            return False
        s = tok.decode([tid])
        st = s.strip()
        if st == "":
            return True
        # punctuation-only
        if re.fullmatch(r"[^\w]+", st):
            return True
        # very short junk fragments (common with BPE)
        if len(st) == 1 and not st.isalnum():
            return True
        return False

    bias: dict[int, float] = {}

    for (phrase, _raw_score), phrase_w in zip(items, w.tolist()):
        phrase = (phrase or "").strip()
        if not phrase:
            continue

        ids = tok(phrase, add_special_tokens=False)["input_ids"]
        if not ids:
            continue

        # Distribute phrase weight across tokens so long phrases don't dominate
        per_tok = float(phrase_w) / max(1, len(ids))

        for tid in ids:
            tid = int(tid)
            if is_junk_token_id(tid):
                continue
            bias[tid] = bias.get(tid, 0.0) + per_tok

    # Optional: cap extreme tokens (prevents single token taking over)
    if per_token_cap is not None and per_token_cap > 0:
        for tid in list(bias.keys()):
            if bias[tid] > per_token_cap:
                bias[tid] = float(per_token_cap)

    return bias


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
        
        self.reader_name = getattr(generator_cfg, "reader_name", None) or "deepset/roberta-base-squad2"
        self.reader_device = getattr(generator_cfg, "reader_device", None) or getattr(self.generator, "device", "cpu")

        self.logger.info("Initializing reader model for candidate mining: %s", self.reader_name)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(self.reader_name, use_fast=True)
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(self.reader_name)
        self.reader_model.eval()
        self.reader_device = "cuda:1" if torch.cuda.is_available() else "cpu"

        # Move reader to device
        if isinstance(self.reader_device, str) and self.reader_device.startswith("cuda"):
            self.reader_model.to(self.reader_device)
            self.logger.info("Moved reader model to device %s", self.reader_device)
        else:
            self.reader_model.to("cpu")
            self.logger.info("Using CPU for reader model") 

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
    

    def _candidates_to_token_bias(
        self,
        candidates: list[tuple[str, float]],
        *,
        max_phrases: int = 40,
        per_token_cap: float = 2.0,
        phrase_score_temperature: float = 1.0,
        drop_junk_tokens: bool = True,
    ) -> dict[int, float]:
        """
        Turn mined candidate strings into sparse token bias dict.

        candidates: list[(text, score)] sorted desc (from _mine_candidates_qa1)
        Returns: {token_id: bias_value} where bias_value is additive to logits (before alpha scaling).

        Design:
        - take top max_phrases candidate strings
        - tokenize each string with GENERATOR tokenizer (not reader tokenizer)
        - distribute the phrase score across its tokens (so multi-token names don't overpower)
        - accumulate per token
        - optionally filter out whitespace/punct tokens

        It biases toward tokens that appear in your mined entities. 
        It does not let "the" dominate because your miner already tries to extract “atomic” spans plus we filter whitespace/punct tokens
        Its much less brittle than doc-prior counting. This is entity/keyword mining -> token IDs -> logit bias
        """
        if not candidates:
            return {}

        if not hasattr(self.generator, "tokenizer"):
            raise RuntimeError("Generator tokenizer not found; need HF generator exposing .tokenizer")

        tok = self.generator.tokenizer
        special_ids = set(getattr(tok, "all_special_ids", []))

        # Use only top phrases
        items = candidates[:max_phrases]

        # Normalize phrase scores so they are stable across queries
        scores = np.array([float(s) for _, s in items], dtype=np.float64)
        # Softmax-ish normalization (temperature)
        scores = scores / max(1e-9, float(phrase_score_temperature))
        scores = scores - scores.max()
        w = np.exp(scores)
        w = w / (w.sum() + 1e-9)  # sums to 1

        def is_junk_token_id(tid: int) -> bool:
            if tid in special_ids:
                return True
            if not drop_junk_tokens:
                return False
            s = tok.decode([tid])
            st = s.strip()
            if st == "":
                return True
            # punctuation-only
            if re.fullmatch(r"[^\w]+", st):
                return True
            # very short junk fragments (common with BPE)
            if len(st) == 1 and not st.isalnum():
                return True
            return False

        bias: dict[int, float] = {}

        for (phrase, _raw_score), phrase_w in zip(items, w.tolist()):
            phrase = (phrase or "").strip()
            if not phrase:
                continue

            ids = tok(phrase, add_special_tokens=False)["input_ids"]
            if not ids:
                continue

            # Distribute phrase weight across tokens so long phrases don't dominate
            per_tok = float(phrase_w) / max(1, len(ids))

            for tid in ids:
                tid = int(tid)
                if is_junk_token_id(tid):
                    continue
                bias[tid] = bias.get(tid, 0.0) + per_tok

        # Optional: cap extreme tokens (prevents single token taking over)
        if per_token_cap is not None and per_token_cap > 0:
            for tid in list(bias.keys()):
                if bias[tid] > per_token_cap:
                    bias[tid] = float(per_token_cap)

        return bias


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
    

    def _mine_candidates_qa1(
        self,
        question: str,
        passages: list[Passage],
        *,
        max_candidates: int = 50,
        per_passage_nbest: int = 8,
        max_answer_chars: int = 80,
        max_seq_len: int = 384,
        doc_stride: int = 128,
        max_span_tokens: int = 8,          # <-- tighter spans
        length_penalty: float = 0.35,      # <-- prefer shorter spans
    ) -> list[tuple[str, float]]:
        """
        Clean reader-based candidate miner (extractive QA + aggressive post-processing).

        Goals:
        - keep candidates atomic (names/dates/numbers/short noun phrases)
        - avoid sentence fragments / multi-clause spans
        - raise candidate quality so LLM selection works again

        Returns: list[(candidate_string, prior_score)] sorted desc, truncated to max_candidates.
        """
        import math
        import re
        import collections
        import torch

        question = (question or "").strip()
        if not question:
            return []

        if not hasattr(self, "reader_model") or self.reader_model is None:
            raise RuntimeError("Reader model not initialized (self.reader_model missing).")

        # -----------------------------
        # Question type (very cheap)
        # -----------------------------
        qlow = question.lower().strip()
        is_who = qlow.startswith("who") or " who " in f" {qlow} "
        is_when = qlow.startswith("when") or " what year" in qlow or " what date" in qlow
        is_where = qlow.startswith("where")
        is_how_many = qlow.startswith("how many") or qlow.startswith("how much")
        # for most NQ-style short answers, this simple typing already helps.

        # -----------------------------
        # Passage relevance weights
        # -----------------------------
        # DPR/IP: Passage.score is similarity (higher better).
        sims = [float(getattr(p, "score", 0.0) or 0.0) for p in passages]
        if sims:
            m = max(sims)
            exps = [math.exp(s - m) for s in sims]
            Z = sum(exps) or 1.0
            sim_w = [e / Z for e in exps]
        else:
            sim_w = [1.0 / max(1, len(passages))] * len(passages)

        rank_w = [1.0 / (1.0 + i) for i in range(len(passages))]
        passage_w = []
        for i in range(len(passages)):
            w = rank_w[i] * (0.2 + 0.8 * sim_w[i] * len(passages))
            passage_w.append(float(w))

        # -----------------------------
        # Cleaning + filtering helpers
        # -----------------------------
        ws_re = re.compile(r"\s+")
        # sentence-ish punctuation we want to avoid in "atomic" answers
        bad_punct_re = re.compile(r"[\.!\?;]")
        # allow commas sometimes (e.g., "Paris, France") but penalize them
        many_commas_re = re.compile(r",.*,")

        year_pat = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
        number_pat = re.compile(r"\b\d+(\.\d+)?\b")
        # common units seen in QA
        unit_pat = re.compile(
            r"\b(pages?|years?|months?|days?|km|kilometers?|miles?|meters?|feet|ft|inches?|%"
            r"|dollars?|usd|euros?|pounds?)\b",
            re.IGNORECASE,
        )
        # Name-like: allow initials and particles
        name_pat = re.compile(
            r"\b([A-Z][a-z]+|[A-Z]\.)"
            r"(?:\s+(?:[A-Z][a-z]+|[A-Z]\.|de|da|del|van|von|al|bin|ibn|la|le|of))*\b"
        )

        def clean_span(s: str) -> str:
            s = (s or "").strip()
            s = ws_re.sub(" ", s)
            s = s.strip(" \t\r\n\"'`.,;:()[]{}")
            return s

        def too_long_or_short(s: str) -> bool:
            if not s:
                return True
            if len(s) < 2:
                return True
            if len(s) > max_answer_chars:
                return True
            toks = s.split()
            if len(toks) == 0:
                return True
            # clamp very long multiword spans
            if len(toks) > max_span_tokens:
                return True
            return False

        def is_junk(s: str) -> bool:
            sl = s.lower()
            if sl in {"the", "a", "an", "it", "they", "he", "she", "this", "that", "these", "those"}:
                return True
            # reject if it looks like a clause/sentence
            if bad_punct_re.search(s):
                return True
            # reject spans with multiple commas (often lists/clauses)
            if many_commas_re.search(s):
                return True
            # reject if mostly non-alnum
            alnum = sum(ch.isalnum() for ch in s)
            if alnum < max(2, int(0.4 * len(s))):
                return True
            return False

        def type_mismatch(s: str) -> bool:
            # Very simple type gating; keep it permissive but helpful.
            if is_who:
                # must contain at least one capitalized token
                return not bool(re.search(r"\b[A-Z][a-z]+\b", s))
            if is_when:
                # must have year or month name or a number (dates)
                return not (bool(year_pat.search(s)) or bool(number_pat.search(s)))
            if is_how_many:
                # must have a number
                return not bool(number_pat.search(s))
            if is_where:
                # locations often capitalized; allow also "in X" style but we won't overfit
                return not bool(re.search(r"\b[A-Z][a-z]+\b", s))
            return False

        def refine_atomic(s: str) -> list[str]:
            """
            If the reader span is still a bit messy, extract better atomic subspans.
            We do NOT return long sentences; we return short entities/dates/numbers.
            """
            out: list[str] = []
            s2 = clean_span(s)
            if not s2 or is_junk(s2):
                return out

            # If question expects a number, prefer number(+unit) chunks
            if is_how_many or is_when:
                # e.g., "581 pages", "1963"
                for m in re.finditer(r"\b\d{1,6}(?:\.\d+)?(?:\s+" + unit_pat.pattern[2:-2] + r")?\b", s2, re.IGNORECASE):
                    out.append(clean_span(m.group(0)))

            # If question expects a year/date, pull years
            if is_when:
                for m in year_pat.findall(s2):
                    out.append(clean_span(m))

            # If question expects a person/org/location, pull name-like phrases
            if is_who or is_where:
                # collect name-like spans, then keep only short ones
                for m in name_pat.findall(s2):
                    c = clean_span(m)
                    if c and 2 <= len(c) <= max_answer_chars and len(c.split()) <= max_span_tokens:
                        out.append(c)

            # Fallback: keep the cleaned original if it is already short/clean
            if not too_long_or_short(s2) and not is_junk(s2):
                out.append(s2)

            # Dedup while preserving order
            seen = set()
            uniq = []
            for x in out:
                k = " ".join(x.lower().split())
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(x)
            return uniq

        # -----------------------------
        # Reader inference + aggregation
        # -----------------------------
        cand_scores: dict[str, float] = collections.defaultdict(float)
        device = next(self.reader_model.parameters()).device

        with torch.no_grad():
            for i, p in enumerate(passages):
                ctx = ((p.title or "") + "\n" + (p.text or ""))[:20000]
                if not ctx.strip():
                    continue

                enc = self.reader_tokenizer(
                    question,
                    ctx,
                    truncation="only_second",
                    max_length=max_seq_len,
                    stride=doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding=False,
                    return_tensors="pt",
                )

                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                offset_mapping = enc["offset_mapping"]

                seq_ids_fn = getattr(enc, "sequence_ids", None)

                out = self.reader_model(input_ids=input_ids, attention_mask=attn)
                start_logits = out.start_logits
                end_logits = out.end_logits

                num_windows = start_logits.size(0)

                for widx in range(num_windows):
                    s_logits = start_logits[widx].detach().cpu()
                    e_logits = end_logits[widx].detach().cpu()

                    if callable(seq_ids_fn):
                        seq_ids = seq_ids_fn(widx)
                    else:
                        seq_ids = [0] * len(offset_mapping[widx])

                    # valid context tokens only
                    valid = []
                    for tidx, off in enumerate(offset_mapping[widx]):
                        if off is None:
                            continue
                        if seq_ids[tidx] == 1 and off[1] > off[0]:
                            valid.append(tidx)
                    if not valid:
                        valid = [tidx for tidx, off in enumerate(offset_mapping[widx]) if off is not None and off[1] > off[0]]
                    if not valid:
                        continue

                    k = max(10, per_passage_nbest * 6)
                    vs = s_logits[valid]
                    ve = e_logits[valid]
                    top_s = torch.topk(vs, k=min(k, vs.numel()))
                    top_e = torch.topk(ve, k=min(k, ve.numel()))

                    top_s_idx = [valid[int(ix)] for ix in top_s.indices.tolist()]
                    top_e_idx = [valid[int(ix)] for ix in top_e.indices.tolist()]

                    spans = []
                    for si in top_s_idx:
                        for ei in top_e_idx:
                            if ei < si:
                                continue
                            span_len = (ei - si) + 1
                            if span_len > max_span_tokens:   # <-- hard cap
                                continue
                            raw_score = float(s_logits[si] + e_logits[ei])
                            adj_score = raw_score - (length_penalty * span_len)  # <-- prefer shorter
                            spans.append((si, ei, adj_score))

                    spans.sort(key=lambda x: x[2], reverse=True)
                    spans = spans[:per_passage_nbest]

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
                        span_text = clean_span(raw)
                        if too_long_or_short(span_text) or is_junk(span_text) or type_mismatch(span_text):
                            # try refining; maybe we can salvage atomic entities/dates/numbers
                            refined = refine_atomic(span_text)
                        else:
                            refined = refine_atomic(span_text)

                        if not refined:
                            continue

                        # Aggregate:
                        # passage weight × (1 + adjusted reader score)
                        # (1+...) keeps it positive-ish; scale is not super important since we normalize later.
                        for c in refined:
                            if too_long_or_short(c) or is_junk(c) or type_mismatch(c):
                                continue
                            cand_scores[c] += passage_w[i] * (1.0 + span_score)

        # -----------------------------
        # Dedup + sort
        # -----------------------------
        def norm_key(s: str) -> str:
            # normalize for merging: lowercase + collapse whitespace + strip surrounding punctuation
            s = s.lower().strip()
            s = ws_re.sub(" ", s)
            s = s.strip(" \t\r\n\"'`.,;:()[]{}")
            return s

        merged: dict[str, float] = collections.defaultdict(float)
        surface: dict[str, str] = {}

        for c, s in cand_scores.items():
            k = norm_key(c)
            if not k:
                continue
            if k not in surface:
                surface[k] = c
            # Keep best surface form preference: longer informative variant for names
            else:
                if len(c) > len(surface[k]) and len(c.split()) <= max_span_tokens:
                    surface[k] = c
            merged[k] += float(s)

        items = sorted(((surface[k], v) for k, v in merged.items()), key=lambda x: x[1], reverse=True)
        return items[:max_candidates]


    def run_logit_rag_stage2(
        self,
        question: str,
        *,
        max_candidates: int = 40,
        max_phrases_for_bias: int = 30,
        alpha: float = 0.8,
        phrase_score_temperature: float = 1.0,
        per_token_cap: float = 2.0,
        clamp_first_line: bool = True,
        hybrid_prompt: bool = False,
    ) -> dict[str, Any]:
        """
        Option B Logit-RAG:
        1) retrieve passages
        2) mine short candidates (entities/dates) using reader QA model
        3) convert candidates -> sparse token bias
        4) generate answer with generator.generate_chat_with_logit_bias

        hybrid_prompt=False means: question-only prompt (pure logit mode)
        hybrid_prompt=True means: include retrieved passages in prompt + also use logit bias (hybrid)
        """

        question = (question or "").strip()
        if not question:
            raise ValueError("question must be non-empty")
        if self.retrieve_only or self.generator is None:
            raise RuntimeError("Stage2 requires generator (retrieve_only=False).")
        if self.top_k <= 0:
            raise ValueError("Stage2 requires retrieval. Set top_k > 0.")

        timings: dict[str, float] = {}
        cache_used = False
        cache_hits = 0
        cache_misses = 0
        bias: dict[int, float] = {}

        # 1) Retrieval
        with timed(timings, "retrieve_total_s"):
            passages, retrieved_doc_ids, rt, cache_info = self._retrieve_passages(question, k=self.top_k)
        timings.update(rt)
        cache_used = bool(cache_info.get("cache_used", False))
        cache_hits = int(cache_info.get("cache_hits", 0))
        cache_misses = int(cache_info.get("cache_misses", 0))

        # 2) Candidate mining
        with timed(timings, "candidate_mine_s"):
            mined = self._mine_candidates_qa1(
                question=question,
                passages=passages,
                max_candidates=max_candidates,
            )

        # mined: list[(candidate_str, score)]
        mined_strings = [c for c, _ in mined]

        # 3) Build sparse token bias from mined candidates
        with timed(timings, "bias_build_s"):
            bias = self._candidates_to_token_bias(
                candidates=mined,
                max_phrases=max_phrases_for_bias,
                per_token_cap=per_token_cap,
                phrase_score_temperature=phrase_score_temperature,
                drop_junk_tokens=True,
            )

        # 4) Build messages
        with timed(timings, "prompt_s"):
            if hybrid_prompt:
                messages, _ = build_rag_messages(question, passages, self.prompt_build_method)
            else:
                # pure logit mode: question only
                messages = [{"role": "user", "content": question}]

        # 5) Generate with logit bias
        with timed(timings, "decode_s"):
            gen = self.generator.generate_chat_with_logit_bias(
                messages=messages,
                bias=bias,
                alpha=float(alpha),
                clamp_first_line=clamp_first_line,
            )

        result = {
            "question": question,
            "answer": gen.text,
            "retrieved_doc_ids": retrieved_doc_ids,
            "mined_candidates": mined_strings[: min(30, len(mined_strings))],
            "bias_tokens": int(len(bias)),
            "alpha": float(alpha),
            "timings_s": timings,
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
            "total_tokens": gen.total_tokens,
            "cache_used": cache_used,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }
        return result





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
            # mined = self._mine_candidates_(passages, max_candidates=max_candidates)
            mined = self._mine_candidates_qa1(
                question=question,
                passages=passages,
                max_candidates=max_candidates)

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
