
from pwrag.evaluator.evaluator import Evaluator
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator, get_refiner, get_judger

from typing import List, Dict, Any
import os
import pandas as pd
from tqdm import tqdm

class BasicPipeline:
    def __init__(self, config, prompt_template=None):
        self.config = config
        self.prompt_template = prompt_template
        self.evaluator = Evaluator(config)
        # self.save_retrieval_cache = config["save_retrieval_cache"]
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template
    
    def run_all(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def run(self, question):
        """The inference process of a single sample."""
        pass

    def evaluate(self, data):
        """Evaluate the generated results."""
        pass

class LLMOnlyPipeline(BasicPipeline):
    """The pipeline runs the generation process without retrieval.
        inference stage: query -> generator
    """
    def __init__(self, config, prompt_template=None, generator=None):
        
        super().__init__(config, prompt_template)
        
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator
    
    def run(self, question, return_dict=False, return_scores=False):
        input_prompts = [self.prompt_template.get_string(question=question)]
        predictions = self.generator.generate(input_prompts, return_dict=return_dict, return_scores=return_scores)
        return predictions
    
    def run_all(self, question_list: List[str]):
        #use tqdm to show the progress
        results = []
        for question in tqdm(question_list):
            predictions = self.run(question)
            results.append(predictions)
        return results
    
class SequentialPipeline(BasicPipeline):
    """The pipeline runs the retrieval, generation and evaluation process sequentially."""
    def __init__(self, config, prompt_template=None, retriever=None, generator=None, cache=None):
        super().__init__(config, prompt_template)
        
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
    
    def run(self, question, return_dict=False, return_scores=False):
        """The inference process of a single sample."""
        # Step 1: Retrieval
        retrieved_docs = self.retriever.search(question)

        # Step 2: Generation
        input_prompts = [self.prompt_template.get_string(question=question, retrieved_docs=retrieved_docs)]
        predictions = self.generator.generate(input_prompts, return_dict=return_dict, return_scores=return_scores)
        
        return predictions


    def naive_run(self, sample, do_eval=True):
        """The inference process of a single sample without RAG."""
        pass
        

