import time
from typing import List
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator, get_refiner, get_judger, timed
from pwrag.dataset.dataset import Item

class BasicPipeline:
    def __init__(self, config, prompt_template=None):
        self.config = config
        self.prompt_template = prompt_template
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run_item(self, question):
        """The inference process of a single sample."""
        pass
   
    def run_batch(self, batch:List[Item]):
        """The inference process of a batch of samples."""
        pass

    def _build_prompt(self, item: Item) -> str:
        t0 = time.perf_counter()
        prompt = self.prompt_template.get_string(question=item.question)
        item.update_perf_metrics("format_prompt_time(s)", time.perf_counter() - t0)
        return prompt


class LLMOnlyPipeline(BasicPipeline):
    """The pipeline runs the generation process without retrieval.
        inference stage: query -> generator
    """
    def __init__(self, config, prompt_template=None, generator=None):
        
        super().__init__(config, prompt_template)

        self.pipeline_name = "LLMOnlyPipeline"
        
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
    
    def run_batch(self, batch:List[Item]):

        input_prompts = [self._build_prompt(item) for item in batch]

        t0 = time.perf_counter()
        predictions, token_info = self.generator.generate(
            input_list=input_prompts,
            return_token_counts=True,)
        #set the generation for item using average time per item in the batch
        item_generation_time = (time.perf_counter() - t0) / len(batch)
        
        for item, pred, p, c, t in zip(
            batch,
            predictions,
            token_info["prompt_token_counts"],
            token_info["completion_token_counts"],
            token_info["total_token_counts"],
        ):
            item.update_output("pred", pred)
            item.update_perf_metrics("generation_time(s)", item_generation_time)
            item.update_perf_metrics("prompt_tokens", int(p))
            item.update_perf_metrics("completion_tokens", int(c))
            item.update_perf_metrics("total_tokens", int(t))
        return batch


class RetrievalOnlyPipeline(BasicPipeline):
    """The pipeline runs the retrieval process without generation.
        inference stage: query -> retriever
    """
    def __init__(self, config, prompt_template=None, retriever=None):
        
        super().__init__(config, prompt_template)
        self.pipeline_name = "RetrievalOnlyPipeline"

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

    def run_batch(self, batch:List[Item]):
        t0 = time.perf_counter()
        input_query = [item.question for item in batch]
        retrieval_results, scores, time_metrics = self.retriever.batch_search(query=input_query,  
                                                                              return_score=True,
                                                                              return_timing_metrics=True)
        #set the generation for item using average time per item in the batch
        item_generation_time = (time.perf_counter() - t0) / len(batch)

        for item, retrieved_docs, score in zip(batch, retrieval_results, scores):
            item.update_output("retrieved_docs", retrieved_docs)
            item.update_output("retrieval_scores", score)
            item.update_perf_metrics("retrieval_time(s)", item_generation_time)
            for k, v in time_metrics.items():
                item.update_perf_metrics(k, v / len(batch))
        return batch
    

class SequentialRAGPipeline(BasicPipeline):

    """The pipeline runs the retrieval, generation and evaluation process sequentially."""
    def __init__(self, config, prompt_template=None,  retriever=None, generator=None, cache=None):
        
        super().__init__(config, prompt_template)

        self.pipeline_name = "SequentialRAGPipeline"

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever
        
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

    def run_item(self, item: Item):
        # Step 1: Retrieval
        metrics: dict[str, float] = {}
        retrieved_docs = self.retriever.search(item.question, metrics=metrics, return_score=False)
        input_prompts = [self.prompt_template.get_string(question=item.question, retrieval_result=retrieved_docs)]
        predictions = self.generator.generate(input_prompts, metrics=metrics)
        item.update_metrics("perf_metrics", metrics)
        return predictions
    
    def run_batch(self, batch: List[Item]):
        perf_metrics: dict[str, float] = {}
       
        with timed(perf_metrics, "total_time(s)"):
            input_query = [item.question for item in batch]
            retrieval_results = self.retriever.batch_search(input_query, metrics=perf_metrics, return_score=False)
            input_prompts = [self.prompt_template.get_string(question=item.question, 
                                                             metrics=perf_metrics,
                                                             retrieval_result=retrieval_result) for item, retrieval_result in zip(batch, retrieval_results)]
            
            pred_answer_list = self.generator.generate(input_prompts, metrics=perf_metrics)
            
            for item, pred in zip(batch, pred_answer_list):
                item.update_output("pred", pred)
        return batch, perf_metrics


