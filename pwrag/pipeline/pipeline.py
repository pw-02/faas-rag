
from pwrag.evaluator.evaluator import Evaluator
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator, get_refiner, get_judger, timed
from typing import List, Dict, Any
import time

class BasicPipeline:
    def __init__(self, config, prompt_template=None):
        self.config = config
        self.prompt_template = prompt_template

        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        
        self.prompt_template = prompt_template

    def run(self, question):
        """The inference process of a single sample."""
        pass

class LLMOnlyPipeline(BasicPipeline):
    """The pipeline runs the generation process without retrieval.
        inference stage: query -> generator
    """
    def __init__(self, config, prompt_template=None, generator=None, return_metrics=False):
        
        super().__init__(config, prompt_template)

        self.pipeline_name = "LLMOnlyPipeline"
        
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

    def run(self, question, return_dict=False, return_scores=False, return_metrics=False):
        metrics: dict[str, float] = {}
        with timed(metrics, "create_prompt(s)"):
            input_prompts = [self.prompt_template.get_string(question=question)]
        
        with timed(metrics, "generation_time(s)"):
            predictions = self.generator.generate(input_prompts, return_dict=return_dict, return_scores=return_scores)
        
        if return_metrics:
            return predictions, metrics
        return predictions
    
class SequentialPipeline(BasicPipeline):

    """The pipeline runs the retrieval, generation and evaluation process sequentially."""
    def __init__(self, config, prompt_template=None, 
                 retriever=None, generator=None, cache=None):
        
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

        if cache is None:
            self.cache = None
        else:
            self.cache = cache
    
    def run(self, question, return_dict=False, return_scores=False, return_metrics=False):
        """The inference process of a single sample."""
        # Step 1: Retrieval
        metrics: dict[str, float] = {}
        retrieval_result = self.retriever.search(question, metrics=metrics, return_score=False)
        with timed(metrics, "create_prompt(s)"):
            input_prompts = [self.prompt_template.get_string(question=question, retrieval_result=retrieval_result)]
        # Step 2: Generation
        with timed(metrics, "generation_time(s)"):
            predictions,usage  = self.generator.generate(input_prompts, 
                                                  return_dict=return_dict, 
                                                  return_scores=return_scores,
                                                  return_usage=True)
            if usage:
                metrics["prompt_tokens"] = usage["prompt_tokens_per_item"][0]
                metrics["completion_tokens"] = usage["completion_tokens_per_item"][0]
                metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
        
        if return_metrics:
            return predictions, metrics
        return predictions