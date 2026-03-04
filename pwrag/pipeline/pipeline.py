import json
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from flask import config

from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.prompt.prompts import get_webpage_to_reasonchain_instruction
from pwrag.utils.utils import get_retriever, get_generator
from pwrag.dataset.dataset import Item
from pwrag.retriever.bing_search import bing_web_search, extract_relevant_info, fetch_page_content, extract_snippet_with_context


@contextmanager
def perf_timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def per_item(elapsed: float, n: int) -> float:
    return elapsed / n if n else 0.0


def default(value, factory):
    return value if value is not None else factory()


class BasicPipeline:
    def __init__(self, config, prompt_template: Optional[PromptTemplate] = None):
        self.config = config
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
 

class Searcho1(BasicPipeline):
    """
    search-01 style web RAG loop (2-stage):
      Stage A: LLM emits <|begin_search_query|>...<|end_search_query|>
      Stage B: pipeline searches + fetches pages, then calls a second prompt to extract/adjust reasoning,
               and appends it back into the prompt/output.
      Repeat until no search query or max_turn reached.
    """

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
        super().__init__(config, prompt_template)

        # retriever unused here (kept for compatibility)
        self.retriever = default(retriever, lambda: get_retriever(config))
        self.generator = default(generator, lambda: get_generator(config))
        self.cache = cache

        # tags
        self.begin_search_query = "<|begin_search_query|>"
        self.end_search_query = "<|end_search_query|>"
        self.begin_search_result = "<|begin_search_result|>"
        self.end_search_result = "<|end_search_result|>"
        self.begin_full_page = "<|begin_full_page|>"
        self.end_full_page = "<|end_full_page|>"

        # stop for stage A generation
        self.stop = [self.end_search_query]

        # limits
        self.max_search_limit = getattr(config, "max_search_limit", 5)
        self.max_turn = getattr(config, "max_turn", 10)
        self.top_k = getattr(config, "top_k", 5)
        self.max_doc_len = getattr(config, "max_doc_len", 2000)

        # web tooling config
        self.bing_subscription_key = getattr(config, "bing_subscription_key", None)
        self.bing_endpoint = getattr(config, "bing_endpoint", "https://api.bing.microsoft.com/v7.0/search")
        self.use_jina = getattr(config, "use_jina", True)
        self.jina_api_key = getattr(config, "jina_api_key", None)

        # generation config
        self.max_tokens = getattr(config, "max_tokens", 2048)

        # in-memory caches
        self.search_cache: Dict[str, Any] = {}  # query -> raw bing result
        self.url_cache: Dict[str, str] = {}     # url -> page content

    def _build_initial_prompt(self, item: Item) -> str:
        return self.prompt_template.get_string(question=item.question, retrieval_result=None)

    @staticmethod
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    def _append(self, item: Item, chunk: str) -> None:
        item.metadata["agent_prompt"] += chunk
        item.metadata["agent_raw_output"] += chunk
        item.metadata["agent_history"].append(chunk)

    # ---------------- Stage A: "reasoning -> emits search query" ----------------
    def run_generation(self, items: List[Item], max_tokens: int):
        """
        Generates ONE turn for each item, stopping on </end_search_query>.
        Adjust kwargs to match your generator backend if needed.
        """
        prompts = [it.metadata["agent_prompt"] for it in items]
        preds, _token_info = self.generator.generate(
            input_list=prompts,
            return_token_counts=False,
            stop=self.stop,
            max_tokens=max_tokens,
        )
        return preds  # list[str], aligned with items

    # ---------------- Optional: same logic as original snippet ----------------
    def replace_recent_steps(self, origin_str: str, replace_str: str) -> str:
        step_pattern = re.compile(r"Step\s+(\d+):\s*")

        def parse_steps(text: str) -> Dict[int, str]:
            steps: Dict[int, str] = {}
            current_step_num = None
            current_content: List[str] = []

            for line in text.splitlines():
                m = step_pattern.match(line)
                if m:
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(m.group(1))
                    rest = line[m.end():].strip()
                    current_content = [rest] if rest else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)

            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            return steps

        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                origin_steps.pop(step_num, None)
            else:
                origin_steps[step_num] = content

        sorted_steps = sorted(origin_steps.items())
        return "\n\n".join([f"{content}" for _, content in sorted_steps])

    # ---------------- Stage B: "web pages -> new reasoning chain" ----------------
    def generate_webpage_to_reasonchain_batch(
        self,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict[str, Any]],
        max_tokens: int,
    ) -> List[str]:
        """
        You MUST adapt this to your repo’s prompt function.
        I’m keeping the interface identical to your original.
        """
        # TODO: replace this with your actual instruction builder
        # user_prompts = [get_webpage_to_reasonchain_instruction(r, sq, doc) ...]
        user_prompts = [
            f"Previous reasoning:\n{r}\n\nSearch query:\n{sq}\n\nDocuments:\n{doc}\n\nUpdate the reasoning."
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]

        # If PromptTemplate should handle chat formatting, use it; otherwise call generator directly.
        preds, _token_info = self.generator.generate(
            input_list=user_prompts,
            return_token_counts=False,
            max_tokens=max_tokens,
        )

        # collect records (like original)
        for p, raw in zip(user_prompts, preds):
            batch_output_records.append({
                "prompt": p,
                "raw_output": raw,
            })

        return preds

    # ---------------- helper: build truncated reasoning like original ----------------
    def _truncate_prev_reasoning(self, full_output: str) -> str:
        lines = full_output.replace("\n\n", "\n").split("\n")
        truncated = ""
        for i, step in enumerate(lines):
            truncated += f"Step {i + 1}: {step}\n\n"

        prev_steps = truncated.split("\n\n")
        if len(prev_steps) <= 5:
            out = "\n\n".join(prev_steps)
        else:
            out = ""
            for i, step in enumerate(prev_steps):
                if i == 0 or i >= len(prev_steps) - 4 or self.begin_search_query in step or self.begin_search_result in step:
                    out += step + "\n\n"
                else:
                    if not out.endswith("\n\n...\n\n"):
                        out += "...\n\n"
        return out.strip("\n")

    def run_batch(self, batch: List[Item]) -> List[Item]:
        if not batch:
            return batch

        # init per-item state
        for item in batch:
            item.update_metadata("agent_executed_search_queries", [])  # list[str]
            item.update_metadata("agent_search_count", 0)
            item.update_metadata("agent_finished", False)
            item.update_metadata("agent_history", [])
            item.update_metadata("agent_raw_output", "")
            item.update_metadata("agent_relevant_info", [])
            item.update_metadata("agent_prompt", self._build_initial_prompt(item))

        batch_output_records: List[Dict[str, Any]] = []
        turn = 0

        while True:
            items_needing_generation = [it for it in batch if not it.metadata["agent_finished"]]
            if not items_needing_generation:
                break

            turn += 1
            if turn > self.max_turn:
                break

            # ----- Stage A: generate -----
            texts = self.run_generation(items_needing_generation, self.max_tokens)

            # per-turn batch collections (must reset EACH turn)
            batch_relevant_info: List[List[Dict[str, Any]]] = []
            batch_original_questions: List[str] = []
            batch_prev_reasonings: List[str] = []
            batch_search_queries: List[str] = []
            batch_documents: List[str] = []
            batch_items: List[Item] = []

            all_urls_to_fetch: set[str] = set()

            # ----- Process each item output, execute search, collect URLs -----
            for item, text in zip(items_needing_generation, texts):
                self._append(item, text)

                search_query = self.extract_between(text, self.begin_search_query, self.end_search_query)

                if not search_query:
                    # no tool call => finished
                    item.metadata["agent_finished"] = True
                    continue

                # enforce search limits / dedupe (original behavior)
                executed = set(item.metadata["agent_executed_search_queries"])
                if item.metadata["agent_search_count"] >= self.max_search_limit:
                    limit_message = (
                        f"\n{self.begin_search_result}\n"
                        "The maximum search limit is exceeded. You are not allowed to search.\n"
                        f"{self.end_search_result}\n"
                    )
                    self._append(item, limit_message)
                    item.metadata["agent_finished"] = True
                    continue

                if search_query in executed:
                    repeat_message = (
                        f"\n{self.begin_search_result}\n"
                        "You have searched this query. Please refer to previous results.\n"
                        f"{self.end_search_result}\n"
                    )
                    self._append(item, repeat_message)
                    item.metadata["agent_finished"] = True
                    continue

                # run/ cache search
                if search_query in self.search_cache:
                    results = self.search_cache[search_query]
                else:
                    try:
                        results = bing_web_search(
                            search_query,
                            self.bing_subscription_key,
                            self.bing_endpoint,
                            market="en-US",
                            language="en",
                        )
                    except Exception:
                        results = {}
                    self.search_cache[search_query] = results

                relevant_info = extract_relevant_info(results)[: self.top_k]
                item.metadata["agent_relevant_info"] = relevant_info

                # collect urls (batch fetch later)
                for info in relevant_info:
                    url = info.get("url")
                    if url and url not in self.url_cache:
                        all_urls_to_fetch.add(url)

                # collect batch stage-B params (original)
                batch_relevant_info.append(relevant_info)
                batch_original_questions.append(item.question)
                batch_prev_reasonings.append(self._truncate_prev_reasoning(item.metadata["agent_raw_output"]))
                batch_search_queries.append(search_query)
                batch_items.append(item)

                # update counters
                item.metadata["agent_search_count"] += 1
                executed.add(search_query)
                item.metadata["agent_executed_search_queries"] = sorted(executed)

            # ----- Batch fetch URLs -----
            if all_urls_to_fetch:
                try:
                    fetched = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=self.use_jina,
                        jina_api_key=self.jina_api_key,
                    )
                except Exception as e:
                    fetched = {u: f"Error fetching URL: {e}" for u in all_urls_to_fetch}

                for url, content in fetched.items():
                    self.url_cache[url] = content

            # ----- Build formatted docs -----
            # NOTE: you still need extract_snippet_with_context in your module
            for relevant_info in batch_relevant_info:
                formatted = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info.get("url", "")
                    raw = self.url_cache.get(url, "")

                    snippet = (doc_info.get("snippet") or "").replace("<b>", "").replace("</b>", "")
                    doc_info["snippet"] = snippet

                    # you likely have this helper already:
                    # success, filtered = extract_snippet_with_context(raw, snippet, context_chars=self.max_doc_len)
                    # context = filtered if success else raw[: self.max_doc_len * 2]
                    context = raw[: self.max_doc_len * 2]  # fallback if helper unavailable

                    doc_info["context"] = context
                    formatted += f"**Web Page {i + 1}:**\n"
                    formatted += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

                batch_documents.append(formatted)

            # ----- Stage B: generate webpage analysis / reasoning update -----
            if batch_items:
                analyses = self.generate_webpage_to_reasonchain_batch(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=getattr(self.config, "dataset_name", "unknown"),
                    batch_output_records=batch_output_records,
                    max_tokens=self.max_tokens,
                )

                for item, analysis in zip(batch_items, analyses):
                    # match original behavior: append inside search_result tags OR replace steps
                    if isinstance(analysis, str):
                        append_text = f"\n\n{self.begin_search_result}{analysis}{self.end_search_result}\n\n"
                        self._append(item, append_text)
                    else:
                        replaced = self.replace_recent_steps(item.metadata["agent_raw_output"], analysis)
                        self._append(item, replaced)

            # loop continues until all finished or max_turn hit

        # final output
        for item in batch:
            item.update_output("pred", item.metadata.get("agent_raw_output", ""))

        # If you want: item.update_output("info_extract_records", batch_output_records)
        return batch     
                        




class AgenticSearchRAG(BasicPipeline):
    """
    Agentic web-RAG loop:
      - model emits <|begin_search_query|>...<|end_search_query|> and/or <|begin_url|>...<|end_url|>
      - pipeline executes tool ops, appends results as <|begin_search_result|> / <|begin_full_page|>
      - repeats until model stops asking for tools or max_turn reached
    """

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
        super().__init__(config, prompt_template)

        # NOTE: retriever is unused here (kept for compatibility / future use)
        self.retriever = default(retriever, lambda: get_retriever(config))
        self.generator = default(generator, lambda: get_generator(config))

        # optional external cache object (if you later want disk caches etc.)
        self.cache = cache

        # tags
        self.begin_search_query = "<|begin_search_query|>"
        self.end_search_query = "<|end_search_query|>"
        self.begin_search_result = "<|begin_search_result|>"
        self.end_search_result = "<|end_search_result|>"
        self.begin_url = "<|begin_url|>"
        self.end_url = "<|end_url|>"
        self.begin_full_page = "<|begin_full_page|>"
        self.end_full_page = "<|end_full_page|>"

        # stops for step generation
        self.stop = [self.end_search_query, self.end_url]

        # limits
        self.max_search_limit = getattr(config, "max_search_limit", 5)
        self.max_url_fetch = getattr(config, "max_url_fetch", 5)
        self.max_turn = getattr(config, "max_turn", 10)
        self.top_k = getattr(config, "top_k", 5)

        # web tooling config
        self.bing_subscription_key = getattr(config, "bing_subscription_key", None)
        self.bing_endpoint = getattr(config, "bing_endpoint", "https://api.bing.microsoft.com/v7.0/search")
        self.use_jina = getattr(config, "use_jina", True)
        self.jina_api_key = getattr(config, "jina_api_key", None)

        # generation config
        self.max_tokens = getattr(config, "max_tokens", 2048)

        # in-memory caches (JSON-serializable values)
        self.search_cache: Dict[str, Any] = {}  # query -> raw bing response (dict)
        self.url_cache: Dict[str, str] = {}     # url -> fetched page content (str)

    @staticmethod
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    def _build_initial_prompt(self, item: Item) -> str:
        # Your PromptTemplate should include the system/instruction formatting you want.
        return self.prompt_template.get_string(question=item.question, retrieval_result=None)

    def _append(self, item: Item, chunk: str) -> None:
        """Append chunk to prompt/history/raw output in metadata."""
        item.metadata["agent_prompt"] += chunk
        item.metadata["agent_raw_output"] += chunk
        item.metadata["agent_history"].append(chunk)

    def _do_search(self, query: str) -> str:
        # cache
        if query in self.search_cache:
            results = self.search_cache[query]
        else:
            results = bing_web_search(
                query,
                self.bing_subscription_key,
                self.bing_endpoint,
                market="en-US",
                language="en",
            )
            self.search_cache[query] = results

        relevant = extract_relevant_info(results)[: self.top_k]
        return json.dumps(relevant, ensure_ascii=False, indent=2)

    def _do_fetch_urls(self, urls: List[str]) -> str:
        fetched: Dict[str, str] = {}
        uncached: List[str] = []

        for u in urls:
            if u in self.url_cache:
                fetched[u] = self.url_cache[u]
            else:
                uncached.append(u)

        if uncached:
            contents = fetch_page_content(uncached, use_jina=self.use_jina, jina_api_key=self.jina_api_key)
            for u, content in contents.items():
                self.url_cache[u] = content
                fetched[u] = content

        return json.dumps(fetched, ensure_ascii=False, indent=2)

    def _step_generate(self, prompts: List[str]) -> List[str]:
        """
        One-step generation for a list of prompts.

        If your generator does NOT support `stop` or `max_tokens`,
        remove those kwargs and rely on generator config instead.
        """
        preds, _token_info = self.generator.generate(
            input_list=prompts,
            return_token_counts=False,
            stop=self.stop,
            max_tokens=self.max_tokens,
        )
        return preds

    def run_batch(self, batch: List[Item]) -> List[Item]:
        if not batch:
            return batch

        # init per-item state (JSON-safe types)
        for item in batch:
            item.update_metadata("agent_pending_operations", [])          # list[dict]
            item.update_metadata("agent_executed_search_queries", [])     # list[str]
            item.update_metadata("agent_executed_url_fetches", [])        # list[str]
            item.update_metadata("agent_search_count", 0)                 # int
            item.update_metadata("agent_finished", False)                 # bool
            item.update_metadata("agent_history", [])                     # list[str]
            item.update_metadata("agent_raw_output", "")                  # str

            # store prompt in metadata so ops can mutate it
            item.update_metadata("agent_prompt", self._build_initial_prompt(item))

        turn = 0
        while True:
            unfinished_items = [it for it in batch if not it.metadata["agent_finished"]]
            if not unfinished_items:
                break

            # 1) execute pending operations first (drain ops before generating)
            items_with_pending_ops = [it for it in unfinished_items if it.metadata["agent_pending_operations"]]
            if items_with_pending_ops:
                for item in items_with_pending_ops:
                    op = item.metadata["agent_pending_operations"].pop(0)  # FIFO
                    op_type = op["type"]
                    op_content = op["content"]

                    if op_type == "search":
                        query = op_content
                        search_str = self._do_search(query)
                        chunk = f"\n{self.begin_search_result}\n{search_str}\n{self.end_search_result}\n"
                        self._append(item, chunk)
                        item.metadata["agent_search_count"] += 1

                    elif op_type == "fetch_url":
                        executed_urls = set(item.metadata["agent_executed_url_fetches"])
                        remaining = self.max_url_fetch - len(executed_urls)
                        if remaining <= 0:
                            chunk = (
                                f"\n{self.begin_full_page}\n"
                                "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
                                f"{self.end_full_page}\n"
                            )
                            self._append(item, chunk)
                            continue  # continue for-loop (next item)

                        urls = [u.strip() for u in op_content.split(",") if u.strip()]
                        urls = [u for u in urls if u not in executed_urls]
                        urls = urls[:remaining]
                        if not urls:
                            continue

                        full_page_str = self._do_fetch_urls(urls)
                        executed_urls.update(urls)
                        # deterministic order
                        item.metadata["agent_executed_url_fetches"] = sorted(executed_urls)

                        chunk = f"\n{self.begin_full_page}\n{full_page_str}\n{self.end_full_page}\n"
                        self._append(item, chunk)

                        # If after this we hit the cap, append the “no more” message like search-01
                        if len(executed_urls) >= self.max_url_fetch:
                            chunk2 = (
                                f"\n{self.begin_full_page}\n"
                                "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
                                f"{self.end_full_page}\n"
                            )
                            self._append(item, chunk2)

                # CRITICAL: restart while-loop; do NOT generate in same iteration
                continue

            # 2) generation step for items without pending ops
            items_needing_generation = [it for it in unfinished_items if not it.metadata["agent_pending_operations"]]
            if not items_needing_generation:
                continue

            turn += 1
            if turn > self.max_turn:
                break

            prompts = [it.metadata["agent_prompt"] for it in items_needing_generation]
            texts = self._step_generate(prompts)

            for item, text in zip(items_needing_generation, texts):
                self._append(item, text)

                search_q = self.extract_between(text, self.begin_search_query, self.end_search_query)
                url_req = self.extract_between(text, self.begin_url, self.end_url)

                executed_search = set(item.metadata["agent_executed_search_queries"])
                executed_urls = set(item.metadata["agent_executed_url_fetches"])
                search_count = int(item.metadata["agent_search_count"])

                if search_q:
                    if search_count < self.max_search_limit and search_q not in executed_search:
                        item.metadata["agent_pending_operations"].append({"type": "search", "content": search_q})
                        executed_search.add(search_q)
                        item.metadata["agent_executed_search_queries"] = sorted(executed_search)
                    else:
                        chunk = (
                            f"\n{self.begin_search_result}\n"
                            "The maximum search limit is exceeded. You are not allowed to search.\n"
                            f"{self.end_search_result}\n"
                        )
                        self._append(item, chunk)

                if url_req:
                    if len(executed_urls) < self.max_url_fetch:
                        urls = [u.strip() for u in url_req.split(",") if u.strip()]
                        urls = [u for u in urls if u not in executed_urls]
                        if urls:
                            item.metadata["agent_pending_operations"].append(
                                {"type": "fetch_url", "content": ", ".join(urls)}
                            )
                    else:
                        chunk = (
                            f"\n{self.begin_full_page}\n"
                            "The maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n"
                            f"{self.end_full_page}\n"
                        )
                        self._append(item, chunk)

                # finished if no tool call emitted
                if not search_q and not url_req:
                    item.metadata["agent_finished"] = True

        # final attach prediction
        for item in batch:
            item.update_output("pred", item.metadata.get("agent_raw_output", ""))

        return batch