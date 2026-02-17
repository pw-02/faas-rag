from __future__ import annotations
import logging
import time
from typing import Any, Dict, List, Optional, Set
import numpy as np
from datetime import datetime, timezone


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
from faasrag.core.prompts import PromptBuildMethodType, build_rag_messages
from faasrag.core.utils import extract_short_answer, append_csv_row
from contextlib import contextmanager
from transformers import LogitsProcessor, LogitsProcessorList
import torch
import re

def is_junk_token(tokenizer, tid: int) -> bool:
    s = tokenizer.decode([tid]).strip().lower()

    # empty or whitespace
    if not s:
        return True

    # punctuation-only
    if re.fullmatch(r"[^\w]+", s):
        return True

    # pure numbers (optional – keep years if you want)
    if re.fullmatch(r"\d+", s):
        return True

    # very short fragments
    if len(s) <= 1:
        return True

    # common stopwords (small starter set)
    if s in {
        "the","a","an","and","or","of","to","in","is","was","for","on","with","as","by"
    }:
        return True

    return False

@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


@torch.no_grad()
def build_doc_log_prior(
    retrieved_chunks: List[str],
    tokenizer,
    mu: float = 1e-3,
    max_length_per_chunk: int = 512,
    ignore_special_tokens: bool = True,
) -> torch.Tensor:
    """
    Build log(p_doc) over vocab from retrieved chunks.

    Returns:
        log_p_doc: FloatTensor [V] on CPU
    """
    vocab_size = tokenizer.vocab_size
    counts = np.zeros(vocab_size, dtype=np.float64)

    special_ids: Set[int] = set()
    if ignore_special_tokens:
        special_ids = set(getattr(tokenizer, "all_special_ids", []))

    for text in retrieved_chunks:
        if not text:
            continue
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length_per_chunk,
            return_attention_mask=False,
        )
        token_ids = enc["input_ids"]
        for tid in token_ids:
            if ignore_special_tokens and tid in special_ids:
                continue
            counts[int(tid)] += 1.0

    counts = counts + float(mu)
    p_doc = counts / counts.sum()
    log_p_doc = torch.tensor(np.log(p_doc), dtype=torch.float32)  # CPU [V]
    return log_p_doc


def log_prior_to_sparse_bias(
    log_p_doc: torch.Tensor,
    top_n: int,
    tokenizer,
    ignore_token_ids: Optional[Set[int]] = None,
    zero_center: bool = True,
) -> Dict[int, float]:
    """
    Convert dense log prior [V] into sparse dict token_id -> log(p_doc[token]).

    We keep only top_n tokens by log-probability.
    Optionally zero-center the log prior to avoid globally depressing logits.
    """
    if log_p_doc is None:
        return {}
    if zero_center:
        log_p_doc = log_p_doc - log_p_doc.mean()

    V = int(log_p_doc.shape[0])
    top_n = int(min(max(top_n, 0), V))
    if top_n == 0:
        return {}

    vals, idx = torch.topk(log_p_doc, k=top_n)  # highest log-prob tokens
    ignore_token_ids = ignore_token_ids or set()

    bias: Dict[int, float] = {}
    for tid, v in zip(idx.tolist(), vals.tolist()):
        if int(tid) in ignore_token_ids:
            continue
        if is_junk_token(tokenizer, tid):
            continue
        bias[int(tid)] = float(v)
    return bias


class SparseAddBiasProcessor(LogitsProcessor):
    """
    Adds alpha * bias[token_id] to logits each step.
    bias is a sparse dict {token_id: bias_value}.
    """
    def __init__(self, bias: Dict[int, float], alpha: float, device):
        self.alpha = float(alpha)
        if not bias:
            self.token_ids = None
            self.bias_vals = None
            return

        self.token_ids = torch.tensor(list(bias.keys()), dtype=torch.long, device=device)
        self.bias_vals = torch.tensor(list(bias.values()), dtype=torch.float32, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.token_ids is None:
            return scores
        # scores: [batch, vocab]
        scores[:, self.token_ids] += self.alpha * self.bias_vals
        return scores


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
        rag_mode: str = "prompt",           # "prompt" or "logit"
        logit_alpha: float = 2.0,           # scaling strength
        logit_top_n: int = 256,             # number of biased tokens
        logit_passage_max_tokens: int = 256 # truncate each passage for bias computation
    ):

        self.logger = logger or logging.getLogger("rag_service")

        self.rag_mode = rag_mode.lower().strip()
        if self.rag_mode not in ("prompt", "logit"):
            raise ValueError("rag_mode must be 'prompt' or 'logit'")
        
        self.logit_alpha = float(logit_alpha)
        self.logit_top_n = int(logit_top_n)
        self.logit_passage_max_tokens = int(logit_passage_max_tokens)

        self.retrieve_only = bool(retrieve_only)
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.always_log_results = bool(always_log_results)
        self.top_k = int(top_k)
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        elif self.top_k == 0 or self.retrieve_only == True:
            self.logger.warning("No retrieval will be performed, pipeline will rely entirely on the generator's prior knowledge.")
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
        else:
            raise ValueError(f"Invalid prompt_build_method {prompt_build_method}")    

        self.max_ctx_chars = int(max_ctx_chars)
        self.seed = seed
        self.cache = None
        self.generator = None

        # 1) Embedder
        self.logger.info("Initializing embedder...")
        self.embedder = build_embedder(embedder_cfg)

        # 2) Index
        self.logger.info("Loading index...")
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)

        # 3) Dim sanity
        self.logger.info("Checking dimension sanity...")
        dim = self.sanity_check_dimensions()

        # 4) Docstore
        self.logger.info("Loading docstore...")
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir, backend=docstore_backend)

        # 5) Cache
        if cache_cfg is not None:
            self.cache = build_cache(cache_cfg, dim=dim, seed=self.seed)

        # 6) Generator
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
    

    # ------------------------------------------------
    # Main entry point (used by gRPC)
    # ------------------------------------------------
    def run(self, question: str) -> dict[str, Any]:
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
        # ALWAYS define bias so result logging never crashes
        bias: dict[int, float] = {}

        # -------------------------
        # Retrieval
        # -------------------------
        if no_retrieval:
            timings.update({"embed_s": 0.0, "ann_s": 0.0, "docstore_s": 0.0})
        else:     
            with timed(timings, "embed_s"):
                qvec = self.embedder.embed_queries([question])
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
                        #thow exception here as something going wrong if index returns a pid that is not in docstore
                        # self.logger.error(f"Docstore missing pid {pid} returned by index. This indicates data inconsistency between index and docstore.")
                        raise ValueError(f"Docstore missing pid {pid} returned by index. This indicates data inconsistency between index and docstore.")
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
        # Prompt / Logit-RAG selection
        # -------------------------
        if self.rag_mode == "prompt":
            # Prompt construction (includes passages)
            with timed(timings, "prompt_s"):
                messages, _ = build_rag_messages(question, passages, self.prompt_build_method)

            with timed(timings, "decode_s"):
                gen = self.generator.generate_chat(messages)

        else:
            # -------------------------
            # Logit-RAG (no text in prompt)
            # -------------------------
            # 1) Build question-only messages
            with timed(timings, "prompt_s"):
                # simplest: only user question, no passages
                # (keeps your chat template behavior)
                # messages = [{"role": "user", "content": question}]
                messages, _ = build_rag_messages(question, passages, self.prompt_build_method)

            # 2) Compute bias ONLINE from retrieved passages
            with timed(timings, "bias_s"):
                # Requires HF-style generator exposing .model and .tokenizer
                if not hasattr(self.generator, "model") or not hasattr(self.generator, "tokenizer"):
                    raise RuntimeError(
                        "Logit-RAG requires generator.model and generator.tokenizer (HF model). "
                        "Your current generator wrapper may be vLLM-only; add a HF path or expose these."
                    )
                retrieved_texts = [p.text for p in passages if p.text]

                log_p_doc  = build_doc_log_prior(
                    retrieved_texts,
                    self.generator.tokenizer,
                    mu=1e-3,
                    max_length_per_chunk=self.logit_passage_max_tokens,
                    ignore_special_tokens=True,
                )
                special_ids = set(getattr(self.generator.tokenizer, "all_special_ids", []))
                
                bias = log_prior_to_sparse_bias(
                    log_p_doc=log_p_doc,
                    top_n=self.logit_top_n,
                    tokenizer=self.generator.tokenizer,
                    ignore_token_ids=special_ids,
                    zero_center=True,
                )

                # print(f"Computed logit bias for {len(bias)} tokens from top-{len(passages)} passages. Sample bias: {list(bias.items())[:10]}")

            # 3) Generate with logits processor
            with timed(timings, "decode_s"):
                if not hasattr(self.generator, "generate_chat_with_logit_bias"):
                    raise RuntimeError(
                        "Please add generator.generate_chat_with_logit_bias(messages, bias, alpha) to your generator."
                    )
                gen = self.generator.generate_chat_with_logit_bias(
                    messages=messages,
                    bias=bias,
                    alpha=self.logit_alpha,
                )
        
        answer = gen.text
        prompt_tokens = gen.prompt_tokens
        completion_tokens = gen.completion_tokens
        total_tokens = gen.total_tokens

        # Add vLLM streaming metrics into timings / metadata
     
        timings["ttft_s"] = gen.metrics.get("ttft_s") or 0.0
        # timings["total_s"] = gen.metrics.get("total_s") or 0.0
        timings["prefill_tps"] = gen.metrics.get("prefill_tps") or 0.0
        timings["decode_tps"] = gen.metrics.get("decode_tps") or 0.0
        finish_reason = gen.metrics.get("finish_reason") or ""
        
        reesult = {
            "question": question,
            "messages": messages,
            "answer": answer, #extract_short_answer(answer, max_chars=20),
            "retrieved_doc_ids": retrieved_doc_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "timings_s": timings,
            "cache_used": cache_used,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "rag_mode": self.rag_mode,
            "logit_alpha": self.logit_alpha if self.rag_mode == "logit" else 0.0,
            "logit_top_n": self.logit_top_n if self.rag_mode == "logit" else 0,
            "bias_tokens": len(bias) if self.rag_mode == "logit" else 0,

        }

        if self.always_log_results:
            self.log_result(reesult)
            
        return reesult
    
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
            "rag_mode": result.get("rag_mode", ""),
            "bias_s": result.get("timings_s", {}).get("bias_s", 0.0),
            "bias_tokens": result.get("bias_tokens", 0),
            "logit_alpha": result.get("logit_alpha", 0.0),
            "logit_top_n": result.get("logit_top_n", 0),

        })
