
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.prompt.prompts import *
from pwrag.utils.utils import get_retriever, get_generator, perf_timer, per_item, default
from pwrag.dataset.dataset import Item
from pwrag.args.args import AppConfig

class BasicPipeline:
    def __init__(self, config, prompt_template: Optional[PromptTemplate] = None):
        self.config:AppConfig = config
        self.prompt_template = default(prompt_template, lambda: PromptTemplate(config))
        self.pipeline_name = self.__class__.__name__

    def run_item(self, question):
        raise NotImplementedError

    def run_batch(self, batch: List[Item]):
        raise NotImplementedError

    def _build_prompt(self, item: Item, *, retrieval_result=None) -> str:
        """Build prompt and record format time. Works for LLM-only and RAG."""
        with perf_timer() as elapsed:
            prompt = self.prompt_template.get_string(
                question=item.question,
                retrieval_result=retrieval_result,
            )
        item.update_perf_metrics("format_prompt_time(s)", elapsed())
        return prompt

    @staticmethod
    def _spread_batch_metrics(items: Sequence[Item], time_metrics: Optional[Dict[str, float]]):
        """Spread batch-level timing metrics across items."""
        if not items or not time_metrics:
            return
        n = len(items)
        for item in items:
            for k, v in time_metrics.items():
                item.update_perf_metrics(k, per_item(v, n))

    @staticmethod
    def _attach_generation_metrics(
        items: Sequence[Item],
        predictions: Sequence[str],
        token_info: Dict[str, Sequence[int]],
        avg_gen_time: float,
        *,
        pred_key: str = "pred",
    ):
        """Attach prediction + token counts + generation time to each item."""
        if not items:
            return

        # safer access (won't KeyError if a backend changes keys)
        p_list = token_info.get("prompt_token_counts", [])
        c_list = token_info.get("completion_token_counts", [])
        t_list = token_info.get("total_token_counts", [])

        for item, pred, p, c, t in zip(items, predictions, p_list, c_list, t_list):
            item.update_output(pred_key, pred)
            item.update_perf_metrics("generation_time(s)", avg_gen_time)
            item.update_perf_metrics("prompt_tokens", int(p))
            item.update_perf_metrics("completion_tokens", int(c))
            item.update_perf_metrics("total_tokens", int(t))


class LLMOnlyPipeline(BasicPipeline):
    """inference stage: query -> generator"""

    def __init__(self, config, prompt_template=None, generator=None):
        super().__init__(config, prompt_template)
        self.generator = default(generator, lambda: get_generator(config))

    def run_batch(self, batch: List[Item]):
        if not batch:
            return batch

        input_prompts = [self._build_prompt(item) for item in batch]

        with perf_timer() as elapsed:
            predictions, token_info = self.generator.generate(
                input_list=input_prompts,
                return_token_counts=True,
            )

        avg_time = per_item(elapsed(), len(batch))
        self._attach_generation_metrics(batch, predictions, token_info, avg_time)
        return batch


class RetrievalOnlyPipeline(BasicPipeline):
    """inference stage: query -> retriever"""

    def __init__(self, config, prompt_template=None, retriever=None):
        super().__init__(config, prompt_template)
        self.retriever = default(retriever, lambda: get_retriever(config))

    def run_batch(self, batch: List[Item]):
        if not batch:
            return batch

        questions = [item.question for item in batch]

        with perf_timer() as elapsed:
            retrieval_results, scores, time_metrics = self.retriever.batch_search(
                query=questions,
                return_score=True,
                return_timing_metrics=True,
            )

        avg_time = per_item(elapsed(), len(batch))
        self._spread_batch_metrics(batch, time_metrics)

        for item, retrieved_result, score in zip(batch, retrieval_results, scores):
            doc_ids = [doc["id"] for doc in retrieved_result]
            item.update_output("retrieval_results", retrieved_result)
            item.update_output("retrieval_doc_ids", doc_ids)
            item.update_output("retrieval_scores", score)
            item.update_perf_metrics("total_retrieval_time(s)", avg_time)

        return batch


class SequentialRAGPipeline(BasicPipeline):
    """inference stage: query -> retriever -> generator (sequential)"""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
        super().__init__(config, prompt_template)

        self.retriever = default(retriever, lambda: get_retriever(config))
        self.generator = default(generator, lambda: get_generator(config))
        self.cache = cache  # kept for compatibility, unused here

    def run_batch(self, batch: List[Item]):
        if not batch:
            return batch

        questions = [item.question for item in batch]

        with perf_timer() as elapsed:
            retrieval_results, time_metrics = self.retriever.batch_search(
                query=questions,
                return_score=False,
                return_timing_metrics=True,
            )

        avg_ret_time = per_item(elapsed(), len(batch))
        self._spread_batch_metrics(batch, time_metrics)

        input_prompts: List[str] = []
        for item, retrieved_result in zip(batch, retrieval_results):
            doc_ids = [doc["id"] for doc in retrieved_result]
            item.update_output("retrieval_doc_ids", doc_ids)
            item.update_perf_metrics("total_retrieval_time(s)", avg_ret_time)
            input_prompts.append(self._build_prompt(item, retrieval_result=retrieved_result))

        with perf_timer() as gen_elapsed:
            predictions, token_info = self.generator.generate(
                input_list=input_prompts,
                return_token_counts=True,
            )

        avg_gen_time = per_item(gen_elapsed(), len(batch))
        self._attach_generation_metrics(batch, predictions, token_info, avg_gen_time)
        return batch
 



# class AgenticSearchRAG(BasicPipeline):
#     """
#     Agentic web-RAG loop:
#       - model emits <|begin_search_query|>...<|end_search_query|> and/or <|begin_url|>...<|end_url|>
#       - pipeline executes tool ops, appends results as <|begin_search_result|> / <|begin_full_page|>
#       - repeats until model stops asking for tools or max_turn reached
#     """

#     def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
#         super().__init__(config, prompt_template)

#         # NOTE: retriever is unused here (kept for compatibility / future use)
#         self.retriever = default(retriever, lambda: get_retriever(config))
#         self.generator = default(generator, lambda: get_generator(config))

#         # optional external cache object (if you later want disk caches etc.)
#         self.cache = cache

#         # tags
#         self.begin_search_query = "<|begin_search_query|>"
#         self.end_search_query = "<|end_search_query|>"
#         self.begin_search_result = "<|begin_search_result|>"
#         self.end_search_result = "<|end_search_result|>"
#         self.begin_url = "<|begin_url|>"
#         self.end_url = "<|end_url|>"
#         self.begin_full_page = "<|begin_full_page|>"
#         self.end_full_page = "<|end_full_page|>"

#         # stops for step generation
#         self.stop = [self.end_search_query, self.end_url]

#         # limits
#         self.max_search_limit = getattr(config, "max_search_limit", 5)
#         self.max_url_fetch = getattr(config, "max_url_fetch", 5)
#         self.max_turn = getattr(config, "max_turn", 10)
#         self.top_k = getattr(config, "top_k", 5)

#         # web tooling config
#         self.bing_subscription_key = getattr(config, "bing_subscription_key", None)
#         self.bing_endpoint = getattr(config, "bing_endpoint", "https://api.bing.microsoft.com/v7.0/search")
#         self.use_jina = getattr(config, "use_jina", True)
#         self.jina_api_key = getattr(config, "jina_api_key", None)

#         # generation config
#         self.max_tokens = getattr(config, "max_tokens", 2048)

#         # in-memory caches (JSON-serializable values)
#         self.search_cache: Dict[str, Any] = {}  # query -> raw bing response (dict)
#         self.url_cache: Dict[str, str] = {}     # url -> fetched page content (str)

#     @staticmethod
#     def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
#         pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
#         matches = re.findall(pattern, text, flags=re.DOTALL)
#         return matches[-1].strip() if matches else None

#     def _build_initial_prompt(self, item: Item) -> str:
#         # Your PromptTemplate should include the system/instruction formatting you want.
#         return self.prompt_template.get_string(question=item.question, retrieval_result=None)

#     def _append(self, item: Item, chunk: str) -> None:
#         """Append chunk to prompt/history/raw output in metadata."""
#         item.metadata["agent_prompt"] += chunk
#         item.metadata["agent_raw_output"] += chunk
#         item.metadata["agent_history"].append(chunk)

#     def _do_search(self, query: str) -> str:
#         # cache
#         if query in self.search_cache:
#             results = self.search_cache[query]
#         else:
#             results = bing_web_search(
#                 query,
#                 self.bing_subscription_key,
#                 self.bing_endpoint,
#                 market="en-US",
#                 language="en",
#             )
#             self.search_cache[query] = results

#         relevant = extract_relevant_info(results)[: self.top_k]
#         return json.dumps(relevant, ensure_ascii=False, indent=2)

#     def _do_fetch_urls(self, urls: List[str]) -> str:
#         fetched: Dict[str, str] = {}
#         uncached: List[str] = []

#         for u in urls:
#             if u in self.url_cache:
#                 fetched[u] = self.url_cache[u]
#             else:
#                 uncached.append(u)

#         if uncached:
#             contents = fetch_page_content(uncached, use_jina=self.use_jina, jina_api_key=self.jina_api_key)
#             for u, content in contents.items():
#                 self.url_cache[u] = content
#                 fetched[u] = content

#         return json.dumps(fetched, ensure_ascii=False, indent=2)

#     def _step_generate(self, prompts: List[str]) -> List[str]:
#         """
#         One-step generation for a list of prompts.

#         If your generator does NOT support `stop` or `max_tokens`,
#         remove those kwargs and rely on generator config instead.
#         """
#         preds, _token_info = self.generator.generate(
#             input_list=prompts,
#             return_token_counts=False,
#             stop=self.stop,
#             max_tokens=self.max_tokens,
#         )
#         return preds

#     def run_batch(self, batch: List[Item]) -> List[Item]:
#         if not batch:
#             return batch

#         # init per-item state (JSON-safe types)
#         for item in batch:
#             item.update_metadata("agent_pending_operations", [])          # list[dict]
#             item.update_metadata("agent_executed_search_queries", [])     # list[str]
#             item.update_metadata("agent_executed_url_fetches", [])        # list[str]
#             item.update_metadata("agent_search_count", 0)                 # int
#             item.update_metadata("agent_finished", False)                 # bool
#             item.update_metadata("agent_history", [])                     # list[str]
#             item.update_metadata("agent_raw_output", "")                  # str

#             # store prompt in metadata so ops can mutate it
#             item.update_metadata("agent_prompt", self._build_initial_prompt(item))

#         turn = 0
#         while True:
#             unfinished_items = [it for it in batch if not it.metadata["agent_finished"]]
#             if not unfinished_items:
#                 break

#             # 1) execute pending operations first (drain ops before generating)
#             items_with_pending_ops = [it for it in unfinished_items if it.metadata["agent_pending_operations"]]
#             if items_with_pending_ops:
#                 for item in items_with_pending_ops:
#                     op = item.metadata["agent_pending_operations"].pop(0)  # FIFO
#                     op_type = op["type"]
#                     op_content = op["content"]

#                     if op_type == "search":
#                         query = op_content
#                         search_str = self._do_search(query)
#                         chunk = f"\n{self.begin_search_result}\n{search_str}\n{self.end_search_result}\n"
#                         self._append(item, chunk)
#                         item.metadata["agent_search_count"] += 1

#                     elif op_type == "fetch_url":
#                         executed_urls = set(item.metadata["agent_executed_url_fetches"])
#                         remaining = self.max_url_fetch - len(executed_urls)
#                         if remaining <= 0:
#                             chunk = (
#                                 f"\n{self.begin_full_page}\n"
#                                 "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
#                                 f"{self.end_full_page}\n"
#                             )
#                             self._append(item, chunk)
#                             continue  # continue for-loop (next item)

#                         urls = [u.strip() for u in op_content.split(",") if u.strip()]
#                         urls = [u for u in urls if u not in executed_urls]
#                         urls = urls[:remaining]
#                         if not urls:
#                             continue

#                         full_page_str = self._do_fetch_urls(urls)
#                         executed_urls.update(urls)
#                         # deterministic order
#                         item.metadata["agent_executed_url_fetches"] = sorted(executed_urls)

#                         chunk = f"\n{self.begin_full_page}\n{full_page_str}\n{self.end_full_page}\n"
#                         self._append(item, chunk)

#                         # If after this we hit the cap, append the “no more” message like search-01
#                         if len(executed_urls) >= self.max_url_fetch:
#                             chunk2 = (
#                                 f"\n{self.begin_full_page}\n"
#                                 "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
#                                 f"{self.end_full_page}\n"
#                             )
#                             self._append(item, chunk2)

#                 # CRITICAL: restart while-loop; do NOT generate in same iteration
#                 continue

#             # 2) generation step for items without pending ops
#             items_needing_generation = [it for it in unfinished_items if not it.metadata["agent_pending_operations"]]
#             if not items_needing_generation:
#                 continue

#             turn += 1
#             if turn > self.max_turn:
#                 break

#             prompts = [it.metadata["agent_prompt"] for it in items_needing_generation]
#             texts = self._step_generate(prompts)

#             for item, text in zip(items_needing_generation, texts):
#                 self._append(item, text)

#                 search_q = self.extract_between(text, self.begin_search_query, self.end_search_query)
#                 url_req = self.extract_between(text, self.begin_url, self.end_url)

#                 executed_search = set(item.metadata["agent_executed_search_queries"])
#                 executed_urls = set(item.metadata["agent_executed_url_fetches"])
#                 search_count = int(item.metadata["agent_search_count"])

#                 if search_q:
#                     if search_count < self.max_search_limit and search_q not in executed_search:
#                         item.metadata["agent_pending_operations"].append({"type": "search", "content": search_q})
#                         executed_search.add(search_q)
#                         item.metadata["agent_executed_search_queries"] = sorted(executed_search)
#                     else:
#                         chunk = (
#                             f"\n{self.begin_search_result}\n"
#                             "The maximum search limit is exceeded. You are not allowed to search.\n"
#                             f"{self.end_search_result}\n"
#                         )
#                         self._append(item, chunk)

#                 if url_req:
#                     if len(executed_urls) < self.max_url_fetch:
#                         urls = [u.strip() for u in url_req.split(",") if u.strip()]
#                         urls = [u for u in urls if u not in executed_urls]
#                         if urls:
#                             item.metadata["agent_pending_operations"].append(
#                                 {"type": "fetch_url", "content": ", ".join(urls)}
#                             )
#                     else:
#                         chunk = (
#                             f"\n{self.begin_full_page}\n"
#                             "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
#                             f"{self.end_full_page}\n"
#                         )
#                         self._append(item, chunk)

#                 # finished if no tool call emitted
#                 if not search_q and not url_req:
#                     item.metadata["agent_finished"] = True

#         # final attach prediction
#         for item in batch:
#             item.update_output("pred", item.metadata.get("agent_raw_output", ""))

#         return batch