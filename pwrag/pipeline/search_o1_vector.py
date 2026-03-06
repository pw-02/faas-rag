from typing import List, Dict, Any, Optional
import re
import json

from pwrag.dataset.dataset import Item
from pwrag.pipeline.pipeline import BasicPipeline
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.prompt.prompts import (
    get_multiqa_search_o1_instruction,
    get_singleqa_search_o1_instruction,
    get_task_instruction_openqa,
)
from pwrag.utils.utils import get_generator, get_retriever


def default(value, factory):
    return value if value is not None else factory()


class SearchO1VectorPipeline(BasicPipeline):
    """
    Agentic RAG pipeline with the same control flow as SearchO1Pipeline,
    but retrieval is done against a vector index / local corpus instead of the web.

    Expected retriever interface:
        retriever.search(query: str, top_k: int) -> List[Dict[str, Any]]

    Each returned result should ideally look like:
        {
            "doc_id": "...",
            "chunk_id": "...",
            "title": "...",
            "text": "...",
            "score": 0.87,
            "metadata": {...}
        }

    Minimum required field:
        - "text"

    Optional config:
        - max_search_limit
        - max_turn
        - top_k
        - max_doc_len
        - max_new_tokens / max_tokens
        - retrieval_query_instruction_fn
        - retrieval_to_reasonchain_instruction_fn
    """

    def __init__(
        self,
        config,
        prompt_template=None,
        retriever=None,
        generator=None,
        cache=None,
    ):
        super().__init__(config, prompt_template)

        self.retriever = default(retriever, lambda: get_retriever(config))
        self.generator = default(generator, lambda: get_generator(config))
        self.cache = cache
        self.prompt_template = default(prompt_template, lambda: PromptTemplate(config))

        # Retrieval tags
        self.begin_search_query = "<|begin_search_query|>"
        self.end_search_query = "<|end_search_query|>"
        self.begin_search_result = "<|begin_search_result|>"
        self.end_search_result = "<|end_search_result|>"

        # Limits / knobs
        self.max_search_limit = getattr(config, "max_search_limit", 5)
        self.max_turn = getattr(config, "max_turn", 10)
        self.top_k = getattr(config, "top_k", 5)
        self.max_doc_len = getattr(config, "max_doc_len", 2000)

        # Generation knobs
        self.max_new_tokens = getattr(config, "max_new_tokens", None)
        if self.max_new_tokens is None:
            self.max_new_tokens = getattr(config, "max_tokens", 8192)

        # In-memory retrieval cache
        self.search_cache: Dict[str, Any] = {}

        # Optional custom instruction builders
        self.retrieval_query_instruction_fn = getattr(
            config,
            "retrieval_query_instruction_fn",
            None,
        )
        self.retrieval_to_reasonchain_instruction_fn = getattr(
            config,
            "retrieval_to_reasonchain_instruction_fn",
            None,
        )

    # -------------------------------------------------------------------------
    # Prompt helpers
    # -------------------------------------------------------------------------

    def _build_initial_prompt(self, item: Item) -> str:
        dataset_name = self.config.dataset.dataset_name.lower()

        if self.retrieval_query_instruction_fn is not None:
            instruction = self.retrieval_query_instruction_fn(self.max_search_limit)
        else:
            # Reuse your existing search-o1 instructions, but replace "web search"
            # language with "knowledge base retrieval" language.
            if any(s in dataset_name for s in ["nq", "naturalquestions"]):
                instruction = get_singleqa_search_o1_instruction(self.max_search_limit)
            elif any(s in dataset_name for s in ["trivia", "triviaqa", "2wiki", "hotpotqa"]):
                instruction = get_multiqa_search_o1_instruction(self.max_search_limit)
            else:
                instruction = get_singleqa_search_o1_instruction(self.max_search_limit)

            instruction = self._convert_web_instruction_to_vector_instruction(instruction)

        user_prompt = get_task_instruction_openqa(item.question)

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = self.prompt_template.get_string(messages=prompt)
        return prompt

    @staticmethod
    def _convert_web_instruction_to_vector_instruction(instruction: str) -> str:
        """
        Light-touch replacement so you can reuse your existing search-o1 prompts.
        """
        replacements = [
            ("web search", "knowledge base retrieval"),
            ("Web search", "Knowledge base retrieval"),
            ("search the web", "retrieve from the knowledge base"),
            ("Search the web", "Retrieve from the knowledge base"),
            ("search engine", "retriever"),
            ("webpage", "retrieved passage"),
            ("web page", "retrieved passage"),
            ("web pages", "retrieved passages"),
            ("search results", "retrieval results"),
            ("Bing", "the retriever"),
            ("browser", "retriever"),
            ("online", "in the knowledge base"),
            ("internet", "knowledge base"),
            ("web", "knowledge base"),
        ]
        out = instruction
        for src, tgt in replacements:
            out = out.replace(src, tgt)
        return out

    def _append(self, item: Item, chunk: str) -> None:
        item.metadata["agent_prompt"] += chunk
        item.metadata["agent_raw_output"] += chunk
        item.metadata["agent_history"].append(chunk)

    # -------------------------------------------------------------------------
    # Parsing helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_between_or_to_end(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        """
        Extract last occurrence of start_tag...end_tag.
        If end_tag is missing, return until end of string.
        """
        if isinstance(text, list):
            if len(text) == 1:
                text = text[0]
            else:
                text = "\n".join(text)

        if not isinstance(text, str):
            return None

        start = text.rfind(start_tag)
        if start == -1:
            return None
        start += len(start_tag)

        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()

        return text[start:end].strip()

    # -------------------------------------------------------------------------
    # Generation helpers
    # -------------------------------------------------------------------------

    def _step_generate(self, prompts: List[str], *, stop: Optional[List[str]] = None) -> List[str]:
        params = {"max_new_tokens": self.max_new_tokens}
        if stop is not None:
            params["stop"] = stop

        outputs = self.generator.generate(
            input_list=prompts,
            return_token_counts=False,
            **params,
        )
        return outputs

    # -------------------------------------------------------------------------
    # Vector retrieval helpers
    # -------------------------------------------------------------------------

    def _do_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Run vector retrieval and return top_k results.
        Uses in-memory caching by query string.
        """
        if query in self.search_cache:
            return self.search_cache[query]

        try:
            results = self.retriever.search(query=query, top_k=self.top_k)
            if results is None:
                results = []
        except Exception as e:
            print(f"Error during vector retrieval for query '{query}': {e}")
            results = []

        # Normalize shape
        normalized = [self._normalize_retrieval_result(r, rank=i) for i, r in enumerate(results)]
        self.search_cache[query] = normalized
        return normalized

    def _normalize_retrieval_result(self, result: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """
        Normalize retriever outputs into a stable schema used by the pipeline.
        """
        if result is None:
            result = {}

        text = result.get("text")
        if text is None:
            # common alternates
            text = (
                result.get("page_content")
                or result.get("content")
                or result.get("chunk")
                or result.get("snippet")
                or ""
            )

        metadata = result.get("metadata", {})
        if metadata is None:
            metadata = {}

        normalized = {
            "source_type": "vector",
            "rank": rank + 1,
            "doc_id": result.get("doc_id", metadata.get("doc_id", "")),
            "chunk_id": result.get("chunk_id", metadata.get("chunk_id", "")),
            "title": result.get("title", metadata.get("title", "")),
            "score": result.get("score", result.get("similarity", None)),
            "text": text,
            "metadata": metadata,
        }
        return normalized

    def _build_documents_for_item(self, relevant_info: List[Dict[str, Any]]) -> str:
        """
        Build formatted retrieval documents for stage-B reasoning.
        """
        formatted_documents = ""

        for i, doc_info in enumerate(relevant_info):
            text = (doc_info.get("text") or "")[: self.max_doc_len]

            payload = {
                "rank": doc_info.get("rank", i + 1),
                "title": doc_info.get("title", ""),
                "doc_id": doc_info.get("doc_id", ""),
                "chunk_id": doc_info.get("chunk_id", ""),
                "score": doc_info.get("score", None),
                "text": text,
                "metadata": doc_info.get("metadata", {}),
            }

            formatted_documents += f"**Retrieved Passage {i + 1}:**\n"
            formatted_documents += json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

        return formatted_documents

    # -------------------------------------------------------------------------
    # Stage-B helpers
    # -------------------------------------------------------------------------

    def _truncate_prev_reasoning_like_search01(self, full_output: str) -> str:
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
        Stage-B: convert retrieved passages into updated reasoning.
        """
        if self.retrieval_to_reasonchain_instruction_fn is None:
            user_prompts = [
                (
                    f"Question: {q}\n\n"
                    f"Previous reasoning:\n{r}\n\n"
                    f"Retrieval query:\n{sq}\n\n"
                    f"Retrieved passages:\n{doc}\n\n"
                    "Update the reasoning using the retrieved evidence only. "
                    "Cite concrete facts from the retrieved passages. "
                    "If the passages are insufficient, say what is still missing."
                )
                for q, r, sq, doc in zip(
                    original_questions,
                    prev_reasonings,
                    search_queries,
                    documents,
                )
            ]
        else:
            user_prompts = [
                self.retrieval_to_reasonchain_instruction_fn(r, sq, doc)
                for r, sq, doc in zip(prev_reasonings, search_queries, documents)
            ]

        preds = self._step_generate(user_prompts, stop=None)

        for p, raw in zip(user_prompts, preds):
            batch_output_records.append({"prompt": p, "raw_output": raw})

        return preds

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run_batch(self, batch: List[Item]) -> List[Item]:
        if not batch:
            return batch

        for item in batch:
            item.update_metadata("agent_prompt", self._build_initial_prompt(item))
            item.update_metadata("agent_raw_output", "")
            item.update_metadata("agent_history", [])
            item.update_metadata("agent_finished", False)

            item.update_metadata("agent_search_count", 0)
            item.update_metadata("agent_executed_search_queries", [])
            item.update_metadata("agent_relevant_info", [])

        batch_output_records: List[Dict[str, Any]] = []
        turn = 0

        while True:
            items_needing_generation = [it for it in batch if not it.metadata["agent_finished"]]
            if not items_needing_generation:
                break

            turn += 1
            if turn > self.max_turn:
                break

            prompts = [it.metadata["agent_prompt"] for it in items_needing_generation]

            stop_list = [self.end_search_query]
            if getattr(self.generator, "tokenizer", None) is not None:
                eos_token = getattr(self.generator.tokenizer, "eos_token", None)
                if eos_token:
                    stop_list.append(eos_token)

            texts = self._step_generate(prompts, stop=stop_list)

            batch_relevant_info: List[List[Dict[str, Any]]] = []
            batch_original_questions: List[str] = []
            batch_prev_reasonings: List[str] = []
            batch_search_queries: List[str] = []
            batch_documents: List[str] = []
            batch_items: List[Item] = []

            for item, text in zip(items_needing_generation, texts):
                self._append(item, text)

                search_query = self.extract_between_or_to_end(
                    text,
                    self.begin_search_query,
                    self.end_search_query,
                )

                if not search_query:
                    item.metadata["agent_finished"] = True
                    continue

                executed = set(item.metadata["agent_executed_search_queries"])
                search_count = int(item.metadata["agent_search_count"])

                if search_count >= self.max_search_limit:
                    limit_message = (
                        f"\n{self.begin_search_result}\n"
                        "The maximum retrieval limit is exceeded. You are not allowed to retrieve more passages.\n"
                        f"{self.end_search_result}\n"
                    )
                    self._append(item, limit_message)
                    item.metadata["agent_finished"] = True
                    continue

                if search_query in executed:
                    repeat_message = (
                        f"\n{self.begin_search_result}\n"
                        "You have already used this retrieval query. Please refer to the previous retrieved passages.\n"
                        f"{self.end_search_result}\n"
                    )
                    self._append(item, repeat_message)
                    item.metadata["agent_finished"] = True
                    continue

                relevant_info = self._do_search(search_query)
                item.metadata["agent_relevant_info"] = relevant_info

                truncated_prev_reasoning = self._truncate_prev_reasoning_like_search01(
                    item.metadata["agent_raw_output"]
                )

                batch_relevant_info.append(relevant_info)
                batch_original_questions.append(item.question)
                batch_prev_reasonings.append(truncated_prev_reasoning)
                batch_search_queries.append(search_query)
                batch_items.append(item)

                item.metadata["agent_search_count"] = search_count + 1
                executed.add(search_query)
                item.metadata["agent_executed_search_queries"] = sorted(executed)

            for relevant_info in batch_relevant_info:
                batch_documents.append(self._build_documents_for_item(relevant_info))

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
                    if isinstance(analysis, str):
                        append_text = (
                            f"\n\n{self.begin_search_result}"
                            f"{analysis}"
                            f"{self.end_search_result}\n\n"
                        )
                        self._append(item, append_text)
                    else:
                        replaced = self.replace_recent_steps(
                            item.metadata["agent_raw_output"],
                            analysis,
                        )
                        self._append(item, replaced)

        for item in batch:
            item.update_output("pred", item.metadata.get("agent_raw_output", ""))

        return batch

    def is_multi_retrival_example(self, item: Item) -> bool:
        item.update_metadata("agent_prompt", self._build_initial_prompt(item))
        prompts = [item.metadata["agent_prompt"]]

        stop_list = [self.end_search_query]
        if getattr(self.generator, "tokenizer", None) is not None:
            eos_token = getattr(self.generator.tokenizer, "eos_token", None)
            if eos_token:
                stop_list.append(eos_token)

        text = self._step_generate(prompts, stop=stop_list)
        search_query = self.extract_between_or_to_end(
            text,
            self.begin_search_query,
            self.end_search_query,
        )
        return bool(search_query)