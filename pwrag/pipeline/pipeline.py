import time
from typing import List
from pwrag.prompt.base_prompt import PromptTemplate
from pwrag.utils.utils import get_retriever, get_generator, get_refiner, get_judger, timed
from pwrag.dataset.dataset import Item, Dataset
import re
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

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



class FLAREPipeline(BasicPipeline):
    def __init__(
        self,
        config,
        threshold=0.2,
        look_ahead_steps=64,
        max_generation_length=256,
        max_iter_num=5,
        prompt_template=None,
        retriever=None,
        generator=None
        ):
        super().__init__(config, prompt_template)
        
        if generator is None:
            generator = get_generator(config)
        if retriever is None:
            retriever = get_retriever(config)
        self.generator = generator
        self.retriever = retriever
        self.pipeline_name = "FLAREPipeline"

        self.threshold = threshold
        self.max_generation_length = max_generation_length
        self.max_iter_num = max_iter_num
        self.look_ahead_steps = look_ahead_steps
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")

    def get_next_sentence(self, output, scores, metrics=None):
        if metrics is None:
            metrics = {}

        with timed(metrics, "get_next_sentence(s)"):        
            tokenizer = self.generator.tokenizer
            text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
            else:
                token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

            output_ids = tokenizer.encode(output, add_special_tokens=False)

            # assert sum([len(s) for s in token_id_sentences]) == len(
            #    output_ids), "token id sentences length not equal to output ids length"
            first_sent_ids = token_id_sentences[0]
            first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score, metrics=None):
        if metrics is None:
            metrics = {}
        
        with timed(metrics, "judge_sent_confidence(s)"):
            judge_result = all([score > self.threshold for score in sent_score])
            new_query = None
            if not judge_result:
                tokenizer = self.generator.tokenizer
                if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    sent_ids = tokenizer.encode(sent, add_special_tokens=False)
                else:
                    sent_ids = tokenizer.encode(sent, allowed_special="all")
                # assert len(sent_ids) == len(sent_score)
                new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.threshold]
                new_query = tokenizer.decode(new_query_ids)
                if len(new_query) == 0:
                    judge_result = True
            return judge_result, new_query
        

    def run_item(self, item: Item, perf_metrics=None):
        if perf_metrics is None:
            perf_metrics = {}

        question = item.question
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:

            input_prompt = self.prompt_template.get_string(question=question, 
                                                           previous_gen=final_gen_result,
                                                           metrics=perf_metrics)

            # input_prompt = self.build_prompt(
            #     question_list=[question], use_reference=False, previous_gen=final_gen_result)[0]
            # scores: token logits of the whole generation seq

            round_gen_output, scores = self.generator.generate(
                input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps,
                metrics=perf_metrics
            )
            
            round_gen_output, scores = round_gen_output[0], scores[0]
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores, metrics=perf_metrics)
            
            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score, metrics=perf_metrics)
            item.update_output(f"judge_result_iter{iter_round}", judge_result)

            if not judge_result:
                # do retrieval-augmented generation
                retrieval_result = self.retriever.search(query, metrics=perf_metrics)
                item.update_output("retrieval_result", retrieval_result)
                input_prompt = self.prompt_template.get_string(
                    question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result, metrics=perf_metrics
                )

                # input_prompt = self.build_prompt(
                #     question_list = [question],
                #     retrieval_results = [retrieval_result],
                #     previous_gen = final_gen_result)[0]
                output, scores = self.generator.generate(
                    input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps,
                    metrics=perf_metrics
                )
                output, scores = output[0], scores[0]
                next_sent, _ = self.get_next_sentence(output, scores, metrics=perf_metrics)
                item.update_output(f"gen_iter_{iter_round}", next_sent)
                item.update_output("retrieval_result", retrieval_result)

            final_gen_result += next_sent
            gen_length += len(next_sent_score)
            iter_round += 1
        
        key = "total_iter_rounds"
        if key in perf_metrics:
            perf_metrics[key] += iter_round
        else:
            perf_metrics[key] = iter_round

        item.update_output("pred", final_gen_result)


    def run_batch(self, batch: List[Item]):
        perf_metrics: dict[str, float] = {}
        with timed(perf_metrics, "total_time(s)"):
            for item in batch:
                self.run_item(item, perf_metrics )
        return batch, perf_metrics