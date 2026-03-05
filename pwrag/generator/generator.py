import time
from typing import Any, Dict, List, Optional
from copy import deepcopy
import warnings
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from tqdm.auto import trange
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from pwrag.args.args import AppConfig
from pwrag.generator.utils import resolve_max_tokens
from pwrag.utils.utils import timed
from pwrag.generator.stop_word_criteria import StopWordCriteria
from vllm import LLM, SamplingParams

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config: AppConfig):
        self._config: AppConfig = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()
    
    def update_base_setting(self):
        self.model_name = self.config.generator.model_name
        self.model_path = self.config.generator.model_path if hasattr(self.config.generator, "model_path") else self.model_name

        self.max_input_len = self.config.generator.max_input_length
        self.batch_size = self.config.generator.batch_size
        self.device = self.config.generator_device if "cuda" in self.config.generator_device and torch.cuda.is_available() else "cpu"
        # self.gpu_num = 0 if self.device == "cpu" else int(self.device.split(":")[1])
        self.gpu_num = 0 if self.device == "cpu" else 1 #default to 1 gpu for vLLM

        # set generation params as a dict not dict config
        self.generation_params = OmegaConf.to_container(self.config.generator.generation_params, resolve=True)
    
    def update_additional_setting(self):
        pass

    def generate(self, input_list: list) -> List[str]:
        """Get responses from the generater.

        Args:
            input_list: it contains input texts, each item represents a sample.

        Returns:
            list: contains generator's response of each input sample.
        """
        pass


class HFCausalLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on hf."""

    def __init__(self, config, model=None):
        super().__init__(config)
        self.model, self.tokenizer = self._load_model(model=model)
        if self.lora_path is not None:
            self.use_lora = True
            self.model.load_adapter(self.lora_path)

    def update_additional_setting(self):
        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            model.to(self.device)
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if "qwen" not in self.model_name:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    def add_new_tokens(self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"):
        import torch
        del self.model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        # get original embedding weight matrix
        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape

        new_tokens_weights = torch.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        # expand vocabulary
        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.tokenizer.add_tokens(new_tokens)

        # create new embedding matrix
        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

        # copy original embeddings to the new weights
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        # append virtual token embeddings to the new weights
        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding

        # update the embedding table
        # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
        embedding_layer.weight.data = new_embedding_weights
        self.model.eval()
        self.model.cuda()
    
    def generate(
        self,
        input_list: List[str],
        batch_size=None,
        return_scores=False,
        return_dict=False,
        return_token_counts: bool = False,
        **params,
    ):
        """Generate batches one by one. The generated content needs to exclude input."""

        if isinstance(input_list, str):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size


        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # generation_params["seed"] = self._config["seed"]
        generation_params["seed"] = None

        # # handle param conflict
        # generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # deal stop params
        stop_sym = None
        if "stop" in generation_params:
            from pwrag.generator.stop_word_criteria import StopWordCriteria

            stop_sym = generation_params.pop("stop")
            stopping_criteria = [
                StopWordCriteria(
                    tokenizer=self.tokenizer,
                    prompts=input_list,
                    stop_words=stop_sym,
                )
            ]
            generation_params["stopping_criteria"] = stopping_criteria

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # set eos token for llama
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in generation_params:
                if isinstance(generation_params["eos_token_id"], int):
                    generation_params["eos_token_id"] = [generation_params["eos_token_id"]]
                generation_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                generation_params["eos_token_id"] = extra_eos_tokens

        responses = []
        scores = []
        generated_token_ids = []
        generated_token_logits = []

        num_prompt_tokens = []
        num_completion_tokens = []

        for idx in trange(0, len(input_list), batch_size, desc="Generation process: ", disable=True):
            with torch.inference_mode():
                torch.cuda.empty_cache()
                batched_prompts = input_list[idx : idx + batch_size]

                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.model.device)

                # Per-example prompt lengths (exclude padding) — IMPORTANT
                prompt_lens = inputs["attention_mask"].sum(dim=1)  # shape [B]

                if return_token_counts:
                    num_prompt_tokens.extend([int(x) for x in prompt_lens.tolist()])

                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_params,
                )

                full_sequences = outputs.sequences  # [B, prompt(padded)+new]

                # extracted_text = st

                # scores (only if you want them; if you want minimal changes, keep as-is)
                logits = torch.stack(outputs.scores, dim=1).softmax(-1)  # [B, steps, vocab]
                # For scoring we need per-example generated ids aligned with steps; we can still use padded prompt len
                padded_prompt_len = inputs["input_ids"].shape[-1]
                gen_ids_for_score = full_sequences[:, padded_prompt_len:]
                steps = min(logits.shape[1], gen_ids_for_score.shape[1])
                gen_score = (
                    torch.gather(logits[:, :steps, :], 2, gen_ids_for_score[:, :steps, None])
                    .squeeze(-1)
                    .cpu()
                    .tolist()
                )
                scores.extend(gen_score)

            # ---- token-counting for completions (per-example slice by real prompt len) ----
            if return_token_counts or return_dict:
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id

                # Build per-example completion token ids using real prompt lens
                batch_completion_ids = []
                batch_completion_counts = []

                for i in range(full_sequences.size(0)):
                    # completion starts after real prompt tokens, not padded length
                    comp_ids = full_sequences[i, prompt_lens[i]:]
                    # remove any left-padding spill if prompt_lens points into padding region (it won't)
                    # count non-pad tokens (safe)
                    comp_count = int((comp_ids != pad_id).sum().item())
                    batch_completion_counts.append(comp_count)
                    batch_completion_ids.append(comp_ids.detach().cpu())

                if return_token_counts:
                    num_completion_tokens.extend(batch_completion_counts)

            # get additional info (tokens/logits) if return_dict
            if return_dict:
                # NOTE: these are ragged sequences; to keep your old behavior (padded to max_new_tokens),
                # we need a dense tensor. We'll pad each to max_new_tokens.
                max_new = generation_params.get("max_new_tokens", None)

                # if max_new_tokens isn't set, fall back to max length in batch
                if max_new is None:
                    max_new = max(x.size(0) for x in batch_completion_ids)

                dense_ids = torch.full(
                    (len(batch_completion_ids), max_new),
                    fill_value=pad_id,
                    dtype=batch_completion_ids[0].dtype,
                )
                for i, ids in enumerate(batch_completion_ids):
                    n = min(ids.size(0), max_new)
                    dense_ids[i, :n] = ids[:n]

                batch_generated_token_logits = (
                    torch.cat([token_scores.unsqueeze(1) for token_scores in outputs.scores], dim=1)
                    .detach()
                    .cpu()
                )
                # If logits shorter than max_new, pad time dimension
                if batch_generated_token_logits.size(1) < max_new:
                    pad_len = max_new - batch_generated_token_logits.size(1)
                    pad_logits = torch.zeros(
                        (batch_generated_token_logits.size(0), pad_len, batch_generated_token_logits.size(-1)),
                        dtype=batch_generated_token_logits.dtype,
                    )
                    batch_generated_token_logits = torch.cat([batch_generated_token_logits, pad_logits], dim=1)
                elif batch_generated_token_logits.size(1) > max_new:
                    batch_generated_token_logits = batch_generated_token_logits[:, :max_new, :]

                generated_token_ids.append(dense_ids)
                generated_token_logits.append(batch_generated_token_logits)

            # decode + strip prompt (keeping your original approach)
            for i, generated_sequence in enumerate(outputs.sequences):
                input_ids = inputs["input_ids"][i]
                text = self.tokenizer.decode(
                    generated_sequence,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                if input_ids is None:
                    prompt_length = 0
                else:
                    prompt_length = len(
                        self.tokenizer.decode(
                            input_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                new_text = text[prompt_length:]

                if stop_sym is not None:
                    lower_stop_index = len(new_text)
                    stops = stop_sym if isinstance(stop_sym, (list, tuple)) else [stop_sym]
                    for sym in stops:
                        stop_index = new_text.find(sym)
                        if stop_index != -1:
                            lower_stop_index = min(stop_index, lower_stop_index)
                    new_text = new_text[:lower_stop_index]

                responses.append(new_text.strip())

        token_info = None
        if return_token_counts:
            total_token_counts = [p + c for p, c in zip(num_prompt_tokens, num_completion_tokens)]
            token_info = {
                "prompt_token_counts": num_prompt_tokens,
                "completion_token_counts": num_completion_tokens,
                "total_token_counts": total_token_counts,
            }

        if return_dict:
            generated_token_ids = torch.cat(generated_token_ids, dim=0) if generated_token_ids else None
            generated_token_logits = torch.cat(generated_token_logits, dim=0) if generated_token_logits else None

            output = {
                "generated_token_ids": generated_token_ids,
                "generated_token_logits": generated_token_logits,
                "responses": responses,
                "scores": scores,
            }
            if return_token_counts:
                output["token_info"] = token_info
            return output

        if return_scores:
            if return_token_counts:
                return responses, scores, token_info
            return responses, scores

        if return_token_counts:
            return responses, token_info

        return responses
    
    
    def cal_gen_probs(self, prev, next):
        import torch
        input_ids = self.tokenizer.encode(prev, add_special_tokens=False)
        target_ids = self.tokenizer.encode(next, add_special_tokens=False)
        context_ids = input_ids + target_ids
        context_tensor = torch.tensor([context_ids]).to(self.device)
        with torch.inference_mode():
            outputs = self.model(context_tensor)
            logits = outputs.logits
            logits = logits[0, len(input_ids) - 1 : len(context_ids) - 1, :]
            logits = logits.to(torch.float32).detach().cpu()
            # softmax to normalize
            probs = torch.softmax(logits, dim=-1)
            # obtain probs of target_ids
            target_probs = probs[range(len(target_ids)), target_ids].numpy()

        return logits, target_probs


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        
        from vllm import LLM
        if self.use_lora:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self.gpu_memory_utilization,
                enable_lora = True,
                max_lora_rank = 64,
                max_logprobs = 32016,
                max_model_len = self.max_model_len
            )
        else:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self.gpu_memory_utilization,
                max_logprobs = 32016,
                max_model_len = self.max_model_len,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # if "llama" in self.model_path.lower():
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "left"
        # Safe for Llama + many decoder-only models

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.98
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self.config.generator.max_input_length

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        return_token_counts=False,
        **params,
    ):
        from vllm import SamplingParams
        import numpy as np

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0

        # generation_params["seed"] = self._config["seed"]
        generation_params["seed"] = None

        # handle param conflict
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # fix for llama3
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        # ---- prompt token counting ----
        if return_token_counts:
            tok = self.tokenizer(
                input_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(self, "max_input_len", None),
            )

            if "attention_mask" in tok:
                prompt_token_counts = tok["attention_mask"].sum(dim=1).tolist()
            else:
                pad_id = self.tokenizer.pad_token_id
                prompt_token_counts = (tok["input_ids"] != pad_id).sum(dim=1).tolist()

        # ---- generation ----
        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params, use_tqdm=False)

        # ---- completion token counting ----
        if return_token_counts:
            completion_token_counts = []
            for output in outputs:
                if not output.outputs:
                    completion_token_counts.append(0)
                    continue

                first = output.outputs[0]
                token_ids = getattr(first, "token_ids", None)

                if token_ids is not None:
                    completion_token_counts.append(len(token_ids))
                else:
                    # fallback
                    completion_token_counts.append(
                        len(self.tokenizer.encode(first.text, add_special_tokens=False))
                    )

            total_token_counts = [
                p + c for p, c in zip(prompt_token_counts, completion_token_counts)
            ]

            token_info = {
                "prompt_token_counts": [int(x) for x in prompt_token_counts],
                "completion_token_counts": [int(x) for x in completion_token_counts],
                "total_token_counts": [int(x) for x in total_token_counts],
            }

        # ---- format outputs ----
        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [
                [c.text for c in output.outputs] if len(output.outputs) > 1 else output.outputs[0].text
                for output in outputs
            ]
            base_output = generated_texts

        # ---- scores ----
        if return_scores:
            scores = []
            for output in outputs:
                output_scores = []
                for single_output in output.outputs:
                    if single_output.logprobs:
                        token_probs = [
                            np.exp(list(score_dict.values())[0].logprob)
                            for score_dict in single_output.logprobs
                        ]
                        output_scores.append(token_probs)
                    else:
                        output_scores.append([])

                scores.append(output_scores[0] if len(output_scores) == 1 else output_scores)

            if return_token_counts:
                return base_output, scores, token_info
            return base_output, scores

        if return_token_counts:
            return base_output, token_info

        return base_output