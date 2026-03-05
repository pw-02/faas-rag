from typing import List, Dict, Any, Optional
import re
import json
from pwrag.dataset.dataset import Item
from pwrag.pipeline.pipeline import BasicPipeline
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.prompt.prompts import get_multiqa_search_o1_instruction, get_singleqa_search_o1_instruction, get_task_instruction_openqa, get_webpage_to_reasonchain_instruction
from pwrag.retriever.bing_search import bing_web_search, extract_relevant_info, fetch_page_content, extract_snippet_with_context
from pwrag.utils.utils import get_retriever, get_generator, perf_timer, per_item, default

def default(value, factory):
    return value if value is not None else factory()


class SearchO1Pipeline(BasicPipeline):
    """
    Search-01 style web RAG (faithful to the original script's control flow):

    Per turn:
      1) For all unfinished items: generate one step, stopping at </end_search_query>
      2) If a search query is emitted:
           - run Bing search (cached)
           - collect top_k URLs
      3) Batch fetch all URLs across items (cached)
      4) Build per-item "documents" (snippets + extracted context)
      5) Stage-B batch generation: "webpage -> reasonchain" (via your instruction builder)
      6) Append Stage-B output back to each item's prompt/output
      7) Repeat until all finished or max_turn reached

    Notes:
      - Uses generator.generate(stop=[end_search_query], max_new_tokens=...)
      - Robustly extracts query even if end tag is stripped by stop handling
      - Stores all state in item.metadata (JSON-safe)
    """

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
        super().__init__(config, prompt_template)

        self.retriever = default(retriever, lambda: get_retriever(config))  # unused (kept)
        self.generator = default(generator, lambda: get_generator(config))
        self.cache = cache  # optional external cache (unused here)
        self.prompt_template = default(prompt_template, lambda: PromptTemplate(config))

        # tags (match search-01)
        self.begin_search_query = "<|begin_search_query|>"
        self.end_search_query = "<|end_search_query|>"
        self.begin_search_result = "<|begin_search_result|>"
        self.end_search_result = "<|end_search_result|>"
        self.begin_full_page = "<|begin_full_page|>"
        self.end_full_page = "<|end_full_page|>"

        # limits / knobs
        self.max_search_limit = getattr(config, "max_search_limit", 5)
        self.max_turn = getattr(config, "max_turn", 10)
        self.top_k = getattr(config, "top_k", 5)
        self.max_doc_len = getattr(config, "max_doc_len", 2000)

        # web config
        self.bing_subscription_key = getattr(config, "bing_subscription_key", None)
        self.bing_endpoint = getattr(config, "bing_endpoint", "https://api.bing.microsoft.com/v7.0/search")
        self.use_jina = getattr(config, "use_jina", True)
        self.jina_api_key = getattr(config, "jina_api_key", None)

        # generation knobs
        # Prefer max_new_tokens (works consistently for HF generator; vLLM wrapper resolves too)
        self.max_new_tokens = getattr(config, "max_new_tokens", None)
        if self.max_new_tokens is None:
            # fall back to prior config naming
            self.max_new_tokens = getattr(config, "max_tokens", 2048)

        # in-memory caches
        self.search_cache: Dict[str, Any] = {}
        self.url_cache: Dict[str, str] = {}

        # Optional: if your repo provides this instruction builder, set it on config or import it.
        # You can inject a callable via config.webpage_to_reasonchain_instruction_fn
        self.webpage_to_reasonchain_instruction_fn = getattr(
            config,
            "webpage_to_reasonchain_instruction_fn",
            None,
        )

    # ---------------- prompt helpers ----------------

    def _build_initial_prompt(self, item: Item) -> str:
        if any(s in self.config.dataset.dataset_name.lower() for s in ['nq', 'naturalquestions']):
            instruction = get_singleqa_search_o1_instruction(self.max_search_limit)
            user_prompt = get_task_instruction_openqa(item.question)
        
        elif any(s in self.config.dataset.dataset_name.lower() for s in ['trivia', 'triviaqa', '2wiki', 'hotpotqa']):
            instruction = get_multiqa_search_o1_instruction(self.max_search_limit)
            user_prompt = get_task_instruction_openqa(item.question)
        
        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = self.prompt_template.get_string(messages=prompt)
        return prompt

        # return self.prompt_template.get_string(question=item.question, retrieval_result=None)
    

    def _append(self, item: Item, chunk: str) -> None:
        item.metadata["agent_prompt"] += chunk
        item.metadata["agent_raw_output"] += chunk
        item.metadata["agent_history"].append(chunk)

    # ---------------- parsing helpers ----------------

    @staticmethod
    def extract_between_or_to_end(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """
        Extract last occurrence of start_tag...end_tag.
        If end_tag is missing (e.g. stripped by stop handling), return to end-of-string.
        """
        start = text.rfind(start_tag)
        if start == -1:
            return None
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    # ---------------- generation helpers ----------------

    def _step_generate(self, prompts: List[str], *, stop: Optional[List[str]] = None) -> List[str]:
        params = {"max_new_tokens": self.max_new_tokens}
        if stop is not None:
            params["stop"] = stop
        # generator.generate returns list[str] for both HF and vLLM wrappers
        outputs = self.generator.generate(
            input_list=prompts,
            return_token_counts=False,
            **params,
        )
        return outputs

    # ---------------- web helpers ----------------

    def _do_search(self, query: str) -> List[Dict[str, Any]]:
        """Return top_k extracted search results (list[dict])."""
        if query in self.search_cache:
            results = self.search_cache[query]
        else:
            try:
                results = bing_web_search(
                    query,
                    self.bing_subscription_key,
                    self.bing_endpoint,
                    market="en-US",
                    language="en",
                )
            except Exception:
                results = {}
            self.search_cache[query] = results

        relevant_info = extract_relevant_info(results)[: self.top_k]
        return relevant_info

    def _batch_fetch_urls(self, urls: List[str]) -> None:
        """Fetch any uncached urls and update url_cache."""
        urls = [u for u in urls if u and u not in self.url_cache]
        if not urls:
            return
        try:
            fetched = fetch_page_content(urls, use_jina=self.use_jina, jina_api_key=self.jina_api_key)
        except Exception as e:
            fetched = {u: f"Error fetching URL: {e}" for u in urls}
        for u, content in fetched.items():
            self.url_cache[u] = content

    # ---------------- stage-B helpers ----------------

    def _truncate_prev_reasoning_like_search01(self, full_output: str) -> str:
        """
        Mirrors the original script's "turn raw output lines into Step i:" truncation.
        """
        all_steps = full_output.replace("\n\n", "\n").split("\n")
        truncated_prev_reasoning = ""
        for i, step in enumerate(all_steps):
            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

        prev_steps = truncated_prev_reasoning.split("\n\n")
        if len(prev_steps) <= 5:
            truncated_prev_reasoning = "\n\n".join(prev_steps)
        else:
            truncated_prev_reasoning = ""
            for i, step in enumerate(prev_steps):
                if (
                    i == 0
                    or i >= len(prev_steps) - 4
                    or self.begin_search_query in step
                    or self.begin_search_result in step
                ):
                    truncated_prev_reasoning += step + "\n\n"
                else:
                    if not truncated_prev_reasoning.endswith("\n\n...\n\n"):
                        truncated_prev_reasoning += "...\n\n"
        return truncated_prev_reasoning.strip("\n")

    def replace_recent_steps(self, origin_str: str, replace_str: str) -> str:
        """
        Same as your original replace_recent_steps.
        """
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
                    rest = line[m.end() :].strip()
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

    def _build_documents_for_item(self, relevant_info: List[Dict[str, Any]]) -> str:
        """
        Build formatted_documents exactly like the original loop, using:
          - snippet (cleaned)
          - context from fetched page (snippet-centered if helper available)
        """
        formatted_documents = ""
        for i, doc_info in enumerate(relevant_info):
            url = doc_info.get("url", "")
            raw_context = self.url_cache.get(url, "")

            snippet = (doc_info.get("snippet") or "").replace("<b>", "").replace("</b>", "")
            doc_info["snippet"] = snippet

            # try to center context around snippet if helper exists
            if extract_snippet_with_context is not None and snippet:
                try:
                    success, filtered_context = extract_snippet_with_context(
                        raw_context,
                        snippet,
                        context_chars=self.max_doc_len,
                    )
                    context = filtered_context if success else raw_context[: self.max_doc_len * 2]
                except Exception:
                    context = raw_context[: self.max_doc_len * 2]
            else:
                context = raw_context[: self.max_doc_len * 2]

            doc_info["context"] = context

            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

        return formatted_documents

    def _stage_b_batch_generate(
        self,
        original_questions: List[str],
        prev_reasonings: List[str],
        search_queries: List[str],
        documents: List[str],
        dataset_name: str,
        batch_output_records: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Mirrors generate_webpage_to_reasonchain_batch, but uses self.generator.
        You MUST provide an instruction builder via:
          - config.webpage_to_reasonchain_instruction_fn, or
          - override this method in a subclass.

        Returns list[str] aligned with inputs.
        """
        if self.webpage_to_reasonchain_instruction_fn is None:
            # Fallback prompt (works, but you should plug your real instruction builder)
            user_prompts = [
                f"Question: {q}\n\nPrevious reasoning:\n{r}\n\nSearch query:\n{sq}\n\nDocuments:\n{doc}\n\n"
                "Update the reasoning and extract the needed information."
                for q, r, sq, doc in zip(original_questions, prev_reasonings, search_queries, documents)
            ]
        else:
            user_prompts = [
                self.webpage_to_reasonchain_instruction_fn(r, sq, doc)
                for r, sq, doc in zip(prev_reasonings, search_queries, documents)
            ]

        preds = self._step_generate(user_prompts, stop=None)

        for p, raw in zip(user_prompts, preds):
            batch_output_records.append({"prompt": p, "raw_output": raw})

        return preds

    # ---------------- main loop ----------------

    def run_batch(self, batch: List[Item]) -> List[Item]:
        if not batch:
            return batch

        # Init per-item state (like active_sequences)
        for item in batch:
            item.update_metadata("agent_prompt", self._build_initial_prompt(item))
            item.update_metadata("agent_raw_output", "")
            item.update_metadata("agent_history", [])
            item.update_metadata("agent_finished", False)

            item.update_metadata("agent_search_count", 0)
            item.update_metadata("agent_executed_search_queries", [])  # JSON-safe list
            item.update_metadata("agent_relevant_info", [])             # last relevant_info (optional)

        batch_output_records: List[Dict[str, Any]] = []
        turn = 0

        while True:
            # sequences_needing_generation = all unfinished items
            items_needing_generation = [it for it in batch if not it.metadata["agent_finished"]]
            if not items_needing_generation:
                break

            turn += 1
            if turn > self.max_turn:
                break

            # ---------- Stage A: reason -> (maybe) emits search query ----------
            prompts = [it.metadata["agent_prompt"] for it in items_needing_generation]
            texts = self._step_generate(prompts, stop=[self.end_search_query, self.generator.tokenizer.eos_token])

            # Per-turn batch collections (reset each turn!)
            batch_relevant_info: List[List[Dict[str, Any]]] = []
            batch_original_questions: List[str] = []
            batch_prev_reasonings: List[str] = []
            batch_search_queries: List[str] = []
            batch_documents: List[str] = []
            batch_items: List[Item] = []

            all_urls_to_fetch: List[str] = []

            for item, text in zip(items_needing_generation, texts):
                # append generation to prompt + output
                self._append(item, text)

                # Extract search query robustly (end tag may be stripped)
                search_query = self.extract_between_or_to_end(
                    text, self.begin_search_query, self.end_search_query
                )

                if not search_query:
                    # no tool call => finished this item (like original else branch)
                    item.metadata["agent_finished"] = True
                    continue

                executed = set(item.metadata["agent_executed_search_queries"])
                search_count = int(item.metadata["agent_search_count"])

                # enforce limits / repeats
                if search_count >= self.max_search_limit:
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

                # search + cache
                relevant_info = self._do_search(search_query)
                item.metadata["agent_relevant_info"] = relevant_info

                # collect urls (uncached) for batch fetch
                for info in relevant_info:
                    url = info.get("url")
                    if url and url not in self.url_cache:
                        all_urls_to_fetch.append(url)

                # build truncated reasoning (like original)
                truncated_prev_reasoning = self._truncate_prev_reasoning_like_search01(
                    item.metadata["agent_raw_output"]
                )

                # collect stage-B params
                batch_relevant_info.append(relevant_info)
                batch_original_questions.append(item.question)
                batch_prev_reasonings.append(truncated_prev_reasoning)
                batch_search_queries.append(search_query)
                batch_items.append(item)

                # update counters
                item.metadata["agent_search_count"] = search_count + 1
                executed.add(search_query)
                item.metadata["agent_executed_search_queries"] = sorted(executed)

            # ---------- Batch fetch URLs ----------
            if all_urls_to_fetch:
                # de-dup while preserving order-ish
                deduped = list(dict.fromkeys(all_urls_to_fetch))
                self._batch_fetch_urls(deduped)

            # ---------- Build formatted docs for stage B ----------
            for relevant_info in batch_relevant_info:
                batch_documents.append(self._build_documents_for_item(relevant_info))

            # ---------- Stage B: webpage -> reasonchain ----------
            if batch_items:
                analyses = self._stage_b_batch_generate(
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=getattr(self.config, "dataset_name", "unknown"),
                    batch_output_records=batch_output_records,
                )

                for item, analysis in zip(batch_items, analyses):
                    # Match original behavior:
                    # - if analysis is a string, wrap it inside search_result tags and append
                    # - else treat it as replacement steps (rarely used; kept for compatibility)
                    if isinstance(analysis, str):
                        append_text = f"\n\n{self.begin_search_result}{analysis}{self.end_search_result}\n\n"
                        self._append(item, append_text)
                    else:
                        replaced = self.replace_recent_steps(item.metadata["agent_raw_output"], analysis)
                        self._append(item, replaced)

            # loop continues until all finished or max_turn hit

        # Final output: match FlashRAG expected output field
        for item in batch:
            item.update_output("pred", item.metadata.get("agent_raw_output", ""))

        # Optional: store batch_output_records for debugging (can be huge)
        # for item in batch:
        #     item.update_output("batch_output_records", batch_output_records)

        return batch