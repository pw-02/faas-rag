from typing import List
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator, get_refiner, get_judger, timed
from pwrag.dataset.dataset import Item, Dataset

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
        perf_metrics: dict[str, float] = {}
        with timed(perf_metrics, "total_time(s)"):
            with timed(perf_metrics, "get_prompts(s)"):
                input_prompts = [self.prompt_template.get_string(question=item.question, metrics=perf_metrics) for item in batch]
            predictions = self.generator.generate(input_prompts, metrics=perf_metrics)
            for item, pred in zip(batch, predictions):
                item.update_output("pred", pred)
        return batch, perf_metrics


    def run_item(self, item: Item):
        perf_metrics: dict[str, float] = {}
        input_prompts = [self.prompt_template.get_string(question=item.question, metrics=perf_metrics)]
        predictions = self.generator.generate(input_prompts, metrics=perf_metrics)
        item.update_metrics("perf_metrics", perf_metrics)
        return predictions
    
    # def run_dataset(self, dataset:Dataset):
    #     perf_metrics: dict[str, float] = {}
    #     with timed(perf_metrics, "total_time(s)"):
    #         input_prompts = [self.prompt_template.get_string(question=q, metrics=perf_metrics) for q in dataset.question]
    #         pred_answer_list = self.generator.generate(input_prompts, metrics=perf_metrics)
    #         dataset.update_output("pred", pred_answer_list)
    #     return dataset, perf_metrics


       
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

    def run_item(self, item: Item):
        perf_metrics: dict[str, float] = {}
        retrieved_docs = self.retriever.search(item.question, metrics=perf_metrics)
        item.update_metrics("perf_metrics", perf_metrics)
        return retrieved_docs
    
    def run_batch(self, batch:List[Item]):
        perf_metrics: dict[str, float] = {}
        
        with timed(perf_metrics, "total_time(s)"):
            input_query = [item.question for item in batch]

            retrieval_results = self.retriever.batch_search(input_query, metrics=perf_metrics, return_score=False)
            for item, retrieved_docs in zip(batch, retrieval_results):
                item.update_output("retrieved_docs", retrieved_docs)
        return batch, perf_metrics
    
    # def run_dataset(self, dataset:Dataset):
    #     perf_metrics: dict[str, float] = {}
    #     with timed(perf_metrics, "total_time(s)"):
    #         input_query = dataset.question
    #         retrieval_results = self.retriever.batch_search(input_query, metrics=perf_metrics)
    #         dataset.update_output("retrieved_docs", retrieval_results)
    #     return dataset, perf_metrics

class SequentialPipeline(BasicPipeline):

    """The pipeline runs the retrieval, generation and evaluation process sequentially."""
    def __init__(self, config, prompt_template=None,  retriever=None, generator=None, cache=None):
        
        super().__init__(config, prompt_template)

        self.pipeline_name = "SequentialPipeline"

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
        input_prompts = [self.prompt_template.get_string(question=item.question, retrieval_result=retrieved_docs, metrics=metrics)]
        predictions = self.generator.generate(input_prompts, metrics=metrics)
        item.update_metrics("perf_metrics", metrics)
        return predictions
    
    def run_batch(self, batch: List[Item]):
        perf_metrics: dict[str, float] = {}
       
        with timed(perf_metrics, "total_time(s)"):
            input_query = [item.question for item in batch]

            retrieval_results = self.retriever.batch_search(input_query, metrics=perf_metrics, return_score=False)

            with timed(perf_metrics, "get_prompts(s)"):
                input_prompts = [self.prompt_template.get_string(question=item.question, retrieval_result=retrieval_result) for item, retrieval_result in zip(batch, retrieval_results)]
            
            pred_answer_list = self.generator.generate(input_prompts, metrics=perf_metrics)
            
            for item, pred in zip(batch, pred_answer_list):
                item.update_output("pred", pred)
        return batch, perf_metrics


    # def run_dataset(self, dataset:Dataset):
    #     perf_metrics: dict[str, float] = {}
    #     with timed(perf_metrics, "total_time(s)"):
    #         input_query = dataset.question
    #         retrieval_results = self.retriever.batch_search(input_query, metrics=perf_metrics, return_score=False)
    #         input_prompts = [self.prompt_template.get_string(question=q, retrieval_result=r, metrics=perf_metrics) for q, r in zip(dataset.question, retrieval_results)]
    #         pred_answer_list = self.generator.generate(input_prompts, metrics=perf_metrics)
    #         dataset.update_output("pred", pred_answer_list)
    #     return dataset, perf_metrics
