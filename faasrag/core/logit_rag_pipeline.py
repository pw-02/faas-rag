from __future__ import annotations
import logging
import time
from typing import Any, Optional
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




@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0
    
@torch.no_grad()
def _passage_to_vector_mean_input_emb(model, tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    """Mean of the model's input embeddings for tokens in the passage -> [H] normalized."""
    device = next(model.parameters()).device
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )["input_ids"].to(device)  # [1, T]
    emb = model.get_input_embeddings()(ids)  # [1, T, H]
    d = emb.mean(dim=1).squeeze(0)           # [H]
    d = d / (d.norm(p=2) + 1e-12)
    return d


@torch.no_grad()
def compute_semantic_logit_bias_from_passages(
    model,
    tokenizer,
    passages: list[Passage],
    top_n: int,
    per_passage_max_tokens: int,
    clamp_negative: bool = True,
) -> dict[int, float]:
    """
    Online semantic bias:
      - build one aggregated passage vector d_agg (mean of passage vectors)
      - score all vocab tokens by cosine similarity to d_agg
      - keep top_n positive scores
    Returns sparse dict: token_id -> score
    """
    device = next(model.parameters()).device
    E = model.get_input_embeddings().weight              # [V, H]
    E_norm = E / (E.norm(dim=1, keepdim=True) + 1e-12)   # [V, H]

    # Aggregate passage vectors
    ds = []
    for p in passages:
        if not p.text:
            continue
        ds.append(_passage_to_vector_mean_input_emb(model, tokenizer, p.text, per_passage_max_tokens))

    if not ds:
        return {}

    d_agg = torch.stack(ds, dim=0).mean(dim=0)           # [H]
    d_agg = d_agg / (d_agg.norm(p=2) + 1e-12)

    # Similarity to all vocab tokens -> [V]
    scores = E_norm @ d_agg

    vals, idx = torch.topk(scores, k=top_n)
    if clamp_negative:
        keep = vals > 0
        vals = vals[keep]
        idx = idx[keep]

    token_ids = idx.detach().cpu().tolist()
    token_vals = vals.detach().cpu().tolist()
    return {int(t): float(v) for t, v in zip(token_ids, token_vals)}

class SparseAddBiasProcessor(LogitsProcessor):
    """Adds alpha * bias[token_id] to logits each step."""
    def __init__(self, bias: dict[int, float], alpha: float, device):
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
                messages = [{"role": "user", "content": question}]

            # 2) Compute bias ONLINE from retrieved passages
            with timed(timings, "bias_s"):
                # Requires HF-style generator exposing .model and .tokenizer
                if not hasattr(self.generator, "model") or not hasattr(self.generator, "tokenizer"):
                    raise RuntimeError(
                        "Logit-RAG requires generator.model and generator.tokenizer (HF model). "
                        "Your current generator wrapper may be vLLM-only; add a HF path or expose these."
                    )
                bias = compute_semantic_logit_bias_from_passages(
                    self.generator.model,
                    self.generator.tokenizer,
                    passages,
                    top_n=self.logit_top_n,
                    per_passage_max_tokens=self.logit_passage_max_tokens,
                    clamp_negative=True,
                )

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
