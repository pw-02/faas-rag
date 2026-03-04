import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator
from pwrag.dataset.dataset import Item


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