
import math
from typing import List
from pwrag.pipeline.pipeline import BasicPipeline
from pwrag.utils.utils import get_retriever, get_generator, timed
from pwrag.dataset.dataset import Item
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
import re
from pwrag.args.args import AppConfig
from pwrag.prompt.base_prompt import PromptTemplate

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


    def run_batch(self, batch_items: List[Item]):
        perf_metrics: dict[str, float] = {}
        with timed(perf_metrics, "total_time(s)"):
            for item in batch_items:
                self.run_item(item, perf_metrics)
        return batch_items, perf_metrics



class RQRAGPipeline(BasicPipeline):
    expand_on_tokens = [
        "[S_Rewritten_Query]",
        "[S_Decomposed_Query]",
        "[S_Disambiguated_Query]",
        "[A_Response]"
    ]
    
    system_prompt = {
        "qa": "Given a question that requires multi-hop reasoning, you need to decompose the question and answer based on the given context. Please provide a short and concise response."
    }
    
    response_generation_params = {
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }
    
    other_generation_params = {
        "temperature": 1,
        "top_p": 0.9,
        "stop": ["[EOS]", "</s>"],
        "skip_special_tokens": False,
        "include_stop_str_in_output": True,
        "logprobs": 1,
        "spaces_between_special_tokens": False,
        "max_tokens": 4096
    }

    def __init__(
        self,
        config: AppConfig,
        prompt_template = None,
        retriever = None,
        generator = None,
        max_depth = 3,
        batch_size = 32
    ):
        super().__init__(config, prompt_template)
        self.pipeline_name = "RQRAGPipeline"
        
        if "chiminchan/rq_rag_llama2_7B" not in config.generator.generator_model:
            print(f"RQRAGPipeline requires generator model 'chiminchan/rq_rag_llama2_7B' from hf, but got {config.generator.generator_model}")
            print(f"Switching to 'chiminchan/rq_rag_llama2_7B'...")
            config.generator.generator_model = "chiminchan/rq_rag_llama2_7B"

        self.generator = generator if generator is not None else get_generator(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.generator.generator_model, padding_side = "left")
        self.retriever = retriever if retriever is not None else get_retriever(config)
        
        self.max_depth = max_depth
        self.batch_size = batch_size
        
        # Due to the low effiency of original method, it only supports vllm now.
    
    def preprocess_eval_data(self, items: List, metrics: dict = None) -> List[str]:

        if metrics is None:
            metrics = {}

        with timed(metrics, "preprocess_eval_data(s)"):
            eval_examples = []
            for item in items:
                eval_example = f"<s><|system|>\n{self.system_prompt['qa']}" + self.tokenizer.eos_token + "\n<|user|>\n" + item.question + self.tokenizer.eos_token + "\n"
                eval_example += "<|assistant|>\n"
                eval_examples.append(eval_example)
            return eval_examples

    def format_evidences(self, evidences: List[str], metrics: dict = None) -> str:
        if metrics is None:
            metrics = {}
        with timed(metrics, "format_evidences(s)"):
            format_evidence = ""
            for evidence in evidences:
                title = evidence['contents'].split('\n')[0]
                text = "\n".join(evidence['contents'].split('\n')[1:])
                format_evidence += f"Title: {title}\n"
                format_evidence += f"Text: {text}\n"
            return format_evidence

    def generate_tree_of_thoughts_batch(self, initial_prompts_batch: List[str], metrics: dict = None):
        if metrics is None:
            metrics = {}
        with timed(metrics, "generate_tree_of_thoughts(s)"):

            paths_batch_dict = {
                idx: [{
                    "prompt": initial_prompt,
                    "depth": 0,
                    "done": False
                }]
                for idx, initial_prompt in enumerate(initial_prompts_batch)
            }
            
            final_outputs_batch = {idx: [] for idx in range(len(initial_prompts_batch))}
            
            while any(paths for paths in paths_batch_dict.values()):
                current_batch = []
                for i, _ in paths_batch_dict.items():
                    if paths_batch_dict[i]:
                        current_path = paths_batch_dict[i].pop(0)
                        current_batch.append(current_path)
                    else:
                        continue
                
                if not current_batch:
                    break
                
                for special_token in self.expand_on_tokens:
                    
                    if current_batch[0]["depth"] >= self.max_depth and special_token != "[A_Response]":
                        continue
                    
                    # Prepare for inputs
                    input_texts = [path["prompt"] + special_token for path in current_batch]
                
                    # Generate outputs
                    if special_token != "[A_Response]":
                        init_outputs = self.generator.generate(
                            input_list = input_texts,
                            return_raw_output = True,
                            metrics=metrics,

                            **self.response_generation_params
                        )
                    else:
                        init_outputs = self.generator.generate(
                            input_list = input_texts,
                            return_raw_output = True,
                            metrics=metrics,
                            **self.other_generation_params
                        )

                    # Decode outputs
                    decoded_outputs = [output.outputs[0].text for output in init_outputs]
                    # Initialize lists to collect queries for batch retrieval
                    queries_for_search = []
                    
                    # Process outputs and prepare for retrieval
                    for i, decoded_output in enumerate(decoded_outputs):
                        current_path = current_batch[i]
                        decoded_output = decoded_output.replace("<s> ", "<s>")
                        
                        if special_token == "[A_Response]":
                            pattern = r"(.*?)\[EOS\]"
                            matches = re.findall(pattern, decoded_output, re.DOTALL)
                            result = matches[-1].strip() if matches else "Unable to detect valid answer"
                            token_ids = init_outputs[i].outputs[0].token_ids[1:-1]
                            logprobs = init_outputs[i].outputs[0].logprobs[1:-1]
                            confidence = 0
                            for token_id, logprobs in zip(token_ids, logprobs):
                                logprob = logprobs[token_id].logprob
                                prob = math.exp(logprob)
                                confidence += prob
                            
                            if len(token_ids) > 0:
                                confidence /= len(token_ids)
                            
                            new_path = {
                                "prompt": input_texts[i] + decoded_output,
                                "depth": current_path["depth"],
                                "done": True,
                                "final_answer": result,
                                "confidence": confidence
                            }
                            final_outputs_batch[i].append(new_path)
                        else:
                            # Extract the query
                            pattern = r"(.*?)\[EOS\]"
                            matches = re.findall(pattern, decoded_output, re.DOTALL)
                            query_for_search = matches[-1].strip() if matches else "dummy"
                            queries_for_search.append(query_for_search)
                    
                    # Perform batch retrieval
                    if queries_for_search:
                        batch_search_results = self.retriever.batch_search(queries_for_search, metrics=metrics)
                        
                        for i, decoded_output in enumerate(decoded_outputs):
                            search_results = batch_search_results[i]
                            format_evidence = self.format_evidences(search_results, metrics=metrics)
                            new_prompt = decoded_output + "[R_Evidences]" + format_evidence + "[/R_Evidences]"
                            new_path = {
                                "prompt": input_texts[i] + new_prompt,
                                "depth": current_path["depth"] + 1,
                                "done": False,
                            }
                            metrics["depth"] = new_path["depth"]
                            paths_batch_dict[i].append(new_path)

            final_outputs_batch_list = [final_outputs_batch[i] for i in range(len(initial_prompts_batch))]
           
            return final_outputs_batch_list

    def select_best_path_single_turn(self, final_outputs, metrics=None):
        # After generating all paths, we can select the best answer
        # Compute perplexity and confidence for each path

        if metrics is None:
            metrics = {}
        with timed(metrics, "select_best_path(s)"):     
            scores = []
            for path in final_outputs:
                confidence = path["confidence"]
                path["confidence"] = confidence
                scores.append((path, confidence))

            # Select the path with the highest confidence
            best_path = max(scores, key = lambda x: x[1])[0]  # x[2] is confidence
            pred = best_path["final_answer"]

            return pred, best_path
    

    def run_batch(self, batch_items: List[Item]):
        perf_metrics: dict[str, float] = {}
        preds = []
        meta_results = []

        with timed(perf_metrics, "total_time(s)"):
            eval_datas = self.preprocess_eval_data(batch_items, metrics=perf_metrics)
            paths_batch = self.generate_tree_of_thoughts_batch(initial_prompts_batch = eval_datas, metrics=perf_metrics)
            for paths in paths_batch:
                pred, best_path = self.select_best_path_single_turn(paths, metrics=perf_metrics)
                preds.append(pred)
                meta_results.append(best_path)
            
            for item, pred in zip(batch_items, preds):
                item.update_output("pred", pred)
            
            for item, meta_result in zip(batch_items, meta_results):
                item.update_output("meta_result", meta_result)

        return batch_items, perf_metrics

