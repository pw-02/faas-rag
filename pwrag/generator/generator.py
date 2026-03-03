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
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoConfig,
)
from pwrag.args.args import AppConfig
from pwrag.generator.utils import resolve_max_tokens
from pwrag.utils.utils import timed
from pwrag.generator.stop_word_criteria import StopWordCriteria

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config: AppConfig):
        self._config = config
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
        self.model_name = self.config.generator.generator_model
        self.model_path = self.config.generator.model_path if hasattr(self.config.generator, "model_path") else self.model_name

        self.max_input_len = self.config.generator.generator_max_input_length
        self.batch_size = self.config.generator.generator_batch_size
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
    """Decoder-only generator based on Hugging Face Transformers, with a vLLM-like return API."""

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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)

        embedding_layer = self.model.get_input_embeddings()
        embedding_weights = embedding_layer.weight
        original_vocab_size, embedding_dim = embedding_weights.shape

        new_tokens_weights = torch.load(token_embedding_path)
        new_tokens_length = new_tokens_weights.shape[0]

        new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
        self.tokenizer.add_tokens(new_tokens)

        new_vocab_size = original_vocab_size + new_tokens_length
        new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim, device=embedding_weights.device, dtype=embedding_weights.dtype)
        new_embedding_weights[:original_vocab_size, :] = embedding_weights

        for token, embedding in zip(new_tokens, new_tokens_weights):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            new_embedding_weights[token_id] = embedding.to(new_embedding_weights.device, dtype=new_embedding_weights.dtype)

        embedding_layer.weight.data = new_embedding_weights
        self.model.eval()
        self.model.cuda()

    @torch.inference_mode()
    def generate(
        self,
        input_list: List[str],
        metrics: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        return_raw_output: bool = False,
        return_scores: bool = False,
        return_dict: bool = False,
        **params,
    ):
        """
        vLLM-like behavior:

        - Default: returns list[str] responses (generated-only, prompt removed)
        - return_raw_output=True: returns HF raw outputs (and still keeps vLLM-like returns depending on flags)
        - return_scores=True: returns (responses, scores) where scores are per-token probabilities of the chosen tokens
        - return_dict=True: returns dict with:
            {
              "responses": list[str],
              "scores": list[list[float]]   # per token probability (chosen token)
              "prompt_tokens_per_item": list[int],
              "completion_tokens_per_item": list[int],
              "generated_token_ids": torch.LongTensor (N, T_max) or None,
              "generated_token_logits": torch.FloatTensor (N, T_max, V) or None,
              "raw_outputs": list[dict] or None   # per-item raw-ish info (optional)
            }
        """
        if metrics is None:
            metrics = {}

        if isinstance(input_list, str):
            input_list = [input_list]

        if batch_size is None:
            batch_size = self.batch_size

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # stop words -> stopping criteria
        stop_sym = None
        if "stop" in generation_params:
            stop_sym = generation_params.pop("stop")
            if isinstance(stop_sym, str):
                stop_sym = [stop_sym]
            generation_params["stopping_criteria"] = [
                StopWordCriteria(tokenizer=self.tokenizer, prompts=input_list, stop_words=stop_sym)
            ]

        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

        # extra eos tokens for llama-like models
        if "llama" in self.model_name.lower():
            extra_eos_tokens = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if "eos_token_id" in generation_params:
                # user might pass int or list
                if isinstance(generation_params["eos_token_id"], int):
                    generation_params["eos_token_id"] = [generation_params["eos_token_id"]]
                generation_params["eos_token_id"].extend(extra_eos_tokens)
            else:
                generation_params["eos_token_id"] = extra_eos_tokens

        responses: List[str] = []
        # vLLMGenerator returns per-token probabilities for the chosen token when return_scores=True
        scores: List[List[float]] = []

        # Optional token-level tensors for return_dict
        generated_token_ids_batches: List[torch.Tensor] = []
        generated_token_logits_batches: List[torch.Tensor] = []

        # Usage-like tracking
        prompt_tokens_per_item: List[int] = []
        completion_tokens_per_item: List[int] = []

        # Optional per-item raw outputs (vLLM-like object list)
        raw_outputs_per_item: List[Dict[str, Any]] = []

        max_new_tokens = generation_params.get("max_new_tokens", None)
        pad_id = self.tokenizer.pad_token_id

        with timed(metrics, "generate(s)"):
            for start in range(0, len(input_list), batch_size):
                batched_prompts = input_list[start : start + batch_size]

                # tokenize
                inputs = self.tokenizer(
                    batched_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_len,
                ).to(self.model.device)

                # prompt token counts (ignore padding)
                if "attention_mask" in inputs:
                    batch_prompt_lens = inputs["attention_mask"].sum(dim=1).to("cpu").tolist()
                else:
                    batch_prompt_lens = (inputs["input_ids"] != pad_id).sum(dim=1).to("cpu").tolist()
                prompt_tokens_per_item.extend(int(x) for x in batch_prompt_lens)

                # generate with raw scores
                # NOTE: output_scores=True returns logits per generated step in outputs.scores
                out = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_params,
                )

                # out.sequences: (B, prompt_padded + gen_len_var)
                # IMPORTANT: slice generated tokens per-example using true prompt lens (not padded length).
                seqs = out.sequences  # (B, L_total)
                bsz = seqs.shape[0]
                gen_ids_list: List[torch.Tensor] = []
                for i in range(bsz):
                    plen = int(batch_prompt_lens[i])
                    gen_ids_list.append(seqs[i, plen:])

                # completion token counts (ignore pad if present)
                # (HF usually doesn't pad completions; still safe)
                for g in gen_ids_list:
                    completion_tokens_per_item.append(int((g != pad_id).sum().item()))

                # Compute per-token probability of chosen token (top-1 for the generated token id)
                # out.scores: list length Tgen, each is (B, vocab)
                # We need align per example because some generations can end early.
                # HF still produces scores for each generated step up to the longest in batch.
                # For each example, use len(gen_ids_list[i]) tokens.
                if out.scores is not None and len(out.scores) > 0:
                    # Stack to (T, B, V) -> (B, T, V)
                    score_stack = torch.stack(out.scores, dim=0).permute(1, 0, 2)  # (B, T, V)
                    prob_stack = score_stack.softmax(dim=-1)

                    batch_scores: List[List[float]] = []
                    for i in range(bsz):
                        g = gen_ids_list[i]
                        t = g.shape[0]
                        if t == 0:
                            batch_scores.append([])
                            continue
                        # gather probs of the generated token ids
                        probs = prob_stack[i, :t, :].gather(1, g[:t].unsqueeze(-1)).squeeze(-1)
                        batch_scores.append(probs.detach().cpu().tolist())
                    scores.extend(batch_scores)
                else:
                    # no scores available
                    scores.extend([[] for _ in range(bsz)])

                # If return_dict, build padded tensors for ids/logits (like your HF code)
                if return_dict:
                    # Decide padding length: prefer max_new_tokens if set, else pad to max generated in this batch
                    pad_to = int(max_new_tokens) if max_new_tokens is not None else max(g.shape[0] for g in gen_ids_list)

                    # token ids padded (B, pad_to)
                    batch_ids = torch.full((bsz, pad_to), fill_value=pad_id, dtype=torch.long)
                    for i, g in enumerate(gen_ids_list):
                        t = min(g.shape[0], pad_to)
                        if t > 0:
                            batch_ids[i, :t] = g[:t].detach().cpu()

                    # logits padded (B, pad_to, V)
                    # out.scores are logits for each step: list[T] of (B,V)
                    if out.scores is not None and len(out.scores) > 0:
                        T = min(len(out.scores), pad_to)
                        V = out.scores[0].shape[-1]
                        batch_logits = torch.zeros((bsz, pad_to, V), dtype=torch.float32)
                        # fill first T steps
                        stacked_logits = torch.stack(out.scores[:T], dim=0).permute(1, 0, 2).detach().cpu().to(torch.float32)
                        batch_logits[:, :T, :] = stacked_logits
                    else:
                        # unknown vocab size if no scores; return None-ish tensor
                        batch_logits = None

                    generated_token_ids_batches.append(batch_ids)
                    if batch_logits is not None:
                        generated_token_logits_batches.append(batch_logits)

                # Decode generated-only text per item
                for i in range(bsz):
                    gen_text = self.tokenizer.decode(
                        gen_ids_list[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    # stop words post-hoc trimming (like your current behavior)
                    if stop_sym is not None and gen_text:
                        cut = len(gen_text)
                        for sym in stop_sym:
                            j = gen_text.find(sym)
                            if j != -1:
                                cut = min(cut, j)
                        gen_text = gen_text[:cut]

                    responses.append(gen_text.strip())

                # Collect raw outputs per item (vLLM-ish)
                if return_raw_output:
                    # For each item, keep the tensors for that item only.
                    # WARNING: logits can be huge; we store references to CPU slices only.
                    # If you want less memory, drop "scores_logits".
                    for i in range(bsz):
                        plen = int(batch_prompt_lens[i])
                        g = gen_ids_list[i].detach()
                        item: Dict[str, Any] = {
                            "prompt": batched_prompts[i],
                            "prompt_len": plen,
                            "sequence_ids": seqs[i].detach().cpu(),         # full (prompt+gen) ids
                            "generated_ids": g.detach().cpu(),              # gen-only ids
                            "generated_text": responses[start + i],
                        }
                        # include per-step logits for this item if available
                        if out.scores is not None and len(out.scores) > 0:
                            # list of (B,V) -> item logits (T,V) on CPU
                            item_logits = torch.stack([s[i].detach().cpu() for s in out.scores], dim=0)
                            item["scores_logits"] = item_logits
                        raw_outputs_per_item.append(item)

            # Metrics like your vLLM path
            perf_info = {
                "prompt_tokens": int(sum(prompt_tokens_per_item)),
                "completion_tokens": int(sum(completion_tokens_per_item)),
                "total_tokens": int(sum(prompt_tokens_per_item) + sum(completion_tokens_per_item)),
            }
            for k, v in perf_info.items():
                metrics[k] = metrics.get(k, 0) + v

        # Assemble return types to match your VLLMGenerator style
        if return_dict:
            gen_ids = torch.cat(generated_token_ids_batches, dim=0) if generated_token_ids_batches else None
            gen_logits = torch.cat(generated_token_logits_batches, dim=0) if generated_token_logits_batches else None

            return {
                "responses": responses,
                "scores": scores,
                "prompt_tokens_per_item": [int(x) for x in prompt_tokens_per_item],
                "completion_tokens_per_item": [int(x) for x in completion_tokens_per_item],
                "generated_token_ids": gen_ids,
                "generated_token_logits": gen_logits,
                "raw_outputs": raw_outputs_per_item if return_raw_output else None,
            }

        if return_scores:
            # vLLMGenerator returns base_output, scores
            base_output = raw_outputs_per_item if return_raw_output else responses
            return base_output, scores

        if return_raw_output:
            return raw_outputs_per_item

        return responses


# class HFCausalLMGenerator(BaseGenerator):
#     """Class for decoder-only generator, based on hf."""

#     def __init__(self, config, model=None):
#         super().__init__(config)
#         self.model, self.tokenizer = self._load_model(model=model)
#         if self.lora_path is not None:
#             self.use_lora = True
#             self.model.load_adapter(self.lora_path)

#     def update_additional_setting(self):
#         self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
#         self.use_lora = False

#     def _load_model(self, model=None):
#         r"""Load model and tokenizer for generator."""
#         if model is None:
#             model = AutoModelForCausalLM.from_pretrained(
#                 self.model_path,
#                 torch_dtype="auto",
#                 device_map=self.device,
#                 trust_remote_code=True,
#             )
#         else:
#             model.to(self.device)
#         model.eval()
#         tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
#         if "qwen" not in self.model_name:
#             tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "left"

#         return model, tokenizer

#     def add_new_tokens(self, token_embedding_path, token_name_func=lambda idx: f"[ref{idx+1}]"):
#         import torch
#         del self.model
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             trust_remote_code=True,
#         )
#         # get original embedding weight matrix
#         embedding_layer = self.model.get_input_embeddings()
#         embedding_weights = embedding_layer.weight
#         original_vocab_size, embedding_dim = embedding_weights.shape

#         new_tokens_weights = torch.load(token_embedding_path)
#         new_tokens_length = new_tokens_weights.shape[0]

#         # expand vocabulary
#         new_tokens = [token_name_func(idx) for idx in range(new_tokens_length)]
#         self.tokenizer.add_tokens(new_tokens)

#         # create new embedding matrix
#         new_vocab_size = original_vocab_size + new_tokens_length
#         new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

#         # copy original embeddings to the new weights
#         new_embedding_weights[:original_vocab_size, :] = embedding_weights

#         # append virtual token embeddings to the new weights
#         for token, embedding in zip(new_tokens, new_tokens_weights):
#             token_id = self.tokenizer.convert_tokens_to_ids(token)
#             new_embedding_weights[token_id] = embedding

#         # update the embedding table
#         # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
#         embedding_layer.weight.data = new_embedding_weights
#         self.model.eval()
#         self.model.cuda()

#     def generate(
#         self,
#         input_list: List[str],
#         metrics: dict[str, float] = None,
#         batch_size=None,
#         return_raw_output=False,
#         return_scores=False,
#         return_dict=False,
#         **params,
#     ):
#         """Generate batches one by one. The generated content needs to exclude input."""

#         if metrics is None:
#             metrics = {}

#         with timed(metrics, "generate(s)"):
#             if isinstance(input_list, str):
#                 input_list = [input_list]
#             if batch_size is None:
#                 batch_size = self.batch_size

#             generation_params = deepcopy(self.generation_params)
#             generation_params.update(params)

#             # deal stop params
#             stop_sym = None
#             if "stop" in generation_params:
#                 stop_sym = generation_params.pop("stop")
#                 stopping_criteria = [
#                     StopWordCriteria(
#                         tokenizer=self.tokenizer,
#                         prompts=input_list,
#                         stop_words=stop_sym,
#                     )
#                 ]
#                 generation_params["stopping_criteria"] = stopping_criteria

#             generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=True)

#             # set eos token for llama
#             if "llama" in self.model_name.lower():
#                 extra_eos_tokens = [
#                     self.tokenizer.eos_token_id,
#                     self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#                 ]
#                 if "eos_token_id" in generation_params:
#                     generation_params["eos_token_id"].extend(extra_eos_tokens)
#                 else:
#                     generation_params["eos_token_id"] = extra_eos_tokens

#             responses = []
#             scores = []
#             generated_token_ids = []
#             generated_token_logits = []
#             # ---- NEW: usage tracking ----
            
#             prompt_tokens_per_item: List[int] = []
#             completion_tokens_per_item: List[int] = []

#             import torch

#             for idx in trange(0, len(input_list), batch_size, desc="Generation process: ", disable=True):
#                 with torch.inference_mode():
#                     torch.cuda.empty_cache()
#                     batched_prompts = input_list[idx : idx + batch_size]
#                     inputs = self.tokenizer(
#                         batched_prompts,
#                         return_tensors="pt",
#                         padding=True,
#                         truncation=True,
#                         max_length=self.max_input_len,
#                     ).to(self.model.device)

#                     outputs = self.model.generate(
#                         **inputs,
#                         output_scores=True,
#                         return_dict_in_generate=True,
#                         **generation_params,
#                     )

#                     # prompt token counts (ignore padding)
#                     if "attention_mask" in inputs:
#                         batch_prompt_counts = inputs["attention_mask"].sum(dim=1).to("cpu").tolist()
#                     else:
#                         pad_id = self.tokenizer.pad_token_id
#                         batch_prompt_counts = (inputs["input_ids"] != pad_id).sum(dim=1).to("cpu").tolist()

#                     # generated token ids: slice off padded prompt length
#                     prompt_len_padded = inputs["input_ids"].shape[-1]
#                     gen_ids = outputs.sequences[:, prompt_len_padded:]  # (B, <=max_new_tokens)

#                     # completion token counts (ignore padding)
#                     pad_id = self.tokenizer.pad_token_id
#                     batch_comp_counts = (gen_ids != pad_id).sum(dim=1).to("cpu").tolist()

#                     prompt_tokens_per_item.extend(int(x) for x in batch_prompt_counts)
#                     completion_tokens_per_item.extend(int(x) for x in batch_comp_counts)

#                     # ---- your scoring logic ----
#                     logits = torch.stack(outputs.scores, dim=1).softmax(-1)
#                     gen_score = torch.gather(logits, 2, gen_ids[:, :, None]).squeeze(-1).cpu().tolist()
#                     scores.extend(gen_score)

#                 # additional info
#                 if return_dict:
#                     batch_generated_token_ids = gen_ids.detach().cpu()
#                     batch_generated_token_logits = (
#                         torch.cat([token_scores.unsqueeze(1) for token_scores in outputs.scores], dim=1)
#                         .detach()
#                         .cpu()
#                     )

#                     # pad to max_new_tokens for uniform shapes (your logic)
#                     if batch_generated_token_ids.shape[1] < generation_params["max_new_tokens"]:
#                         real_batch_size, num_generated_tokens = batch_generated_token_ids.shape
#                         padding_length = generation_params["max_new_tokens"] - num_generated_tokens
#                         padding_token_ids = torch.full(
#                             (real_batch_size, padding_length),
#                             fill_value=self.tokenizer.pad_token_id,
#                             dtype=batch_generated_token_ids.dtype,
#                         )
#                         padding_token_logits = torch.zeros(
#                             (real_batch_size, padding_length, batch_generated_token_logits.shape[-1]),
#                             dtype=batch_generated_token_logits.dtype,
#                         )
#                         batch_generated_token_ids = torch.cat([batch_generated_token_ids, padding_token_ids], dim=1)
#                         batch_generated_token_logits = torch.cat([batch_generated_token_logits, padding_token_logits], dim=1)

#                     generated_token_ids.append(batch_generated_token_ids)
#                     generated_token_logits.append(batch_generated_token_logits)

#                 # ---- IMPORTANT CHANGE: decode only generated part (correct) ----
#                 for i in range(gen_ids.shape[0]):
#                     gen_text = self.tokenizer.decode(
#                         gen_ids[i],
#                         skip_special_tokens=True,
#                         clean_up_tokenization_spaces=False,
#                     )

#                     # apply stop words post-hoc to the generated text
#                     if stop_sym is not None:
#                         lower_stop_index = len(gen_text)
#                         for sym in stop_sym:
#                             stop_index = gen_text.find(sym)
#                             if stop_index != -1:
#                                 lower_stop_index = min(stop_index, lower_stop_index)
#                         gen_text = gen_text[:lower_stop_index]

#                     responses.append(gen_text.strip())

           
            
#             perf_info = {
#                 "prompt_tokens": sum(prompt_tokens_per_item),
#                 "completion_tokens": sum(completion_tokens_per_item),
#                 "total_tokens": sum(prompt_tokens_per_item) + sum(completion_tokens_per_item),
#             }
#             for key in perf_info.keys():
#                 if key not in metrics:
#                     metrics[key] = perf_info[key]
#                 else:
#                     metrics[key] += perf_info[key]

#             if return_dict:
#                 generated_token_ids = torch.cat(generated_token_ids, dim=0) if generated_token_ids else None
#                 generated_token_logits = torch.cat(generated_token_logits, dim=0) if generated_token_logits else None
#                 return {
#                     "generated_token_ids": generated_token_ids,
#                     "generated_token_logits": generated_token_logits,
#                     "responses": responses,
#                     "scores": scores,
#                 }
            
#             if return_scores:
#                 return responses, scores
#             else:
#                 return responses
            

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
                max_model_len = self.max_model_len
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
    
    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            print(f"Using {self.gpu_num} GPUs for tensor parallelism.")
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = None if "generator_lora_path" not in self._config else self._config["generator_lora_path"]
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self.config.generator.generator_max_input_length



    def generate(
        self,
        input_list: List[str],
        metrics: dict[str, float] = None,
        return_raw_output: bool = False,
        return_scores: bool = False,
        return_dict: bool = False,
        **params,
    ):
        """
        vLLM generate that matches HFCausalLMGenerator's metric collection:
        - metrics["prompt_tokens"]
        - metrics["completion_tokens"]
        - metrics["total_tokens"]
        - metrics["generation(s)"]
        """
        from vllm import SamplingParams
        import numpy as np
        import time

        if metrics is None:
            metrics = {}

        t0 = time.perf_counter()

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        # HF compatibility: do_sample=False => temperature=0
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0

        generation_params["seed"] = self._config["seed"]

        # handle param conflict / max tokens
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # fix for llama3 / stop tokens
        if "stop" in generation_params:
            # be robust if user passed a string
            if isinstance(generation_params["stop"], str):
                generation_params["stop"] = [generation_params["stop"]]
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        # If returning scores, ensure logprobs are requested.
        # NOTE: vLLM logprobs are per generated token.
        if return_scores and "logprobs" not in generation_params:
            generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        # ---- prompt token accounting (like HF) ----
        # Count prompt tokens without padding.
        # Works for any tokenizer; prefer attention_mask if available.
        tok = self.tokenizer(
            input_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self, "max_input_len", None) or getattr(self, "max_model_len", None),
        )
        if "attention_mask" in tok:
            prompt_tokens_per_item = tok["attention_mask"].sum(dim=1).tolist()
        else:
            pad_id = self.tokenizer.pad_token_id
            prompt_tokens_per_item = (tok["input_ids"] != pad_id).sum(dim=1).tolist()

        # ---- run vLLM ----
        if self.use_lora:
            from vllm.lora.request import LoRARequest
            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params)

        # ---- completion token accounting ----
        # Prefer vLLM-provided token_ids if present (most reliable).
        completion_tokens_per_item = []
        for req_out in outputs:
            # req_out.outputs can have >1 candidate if n>1 / best_of, etc.
            # We mirror your current behavior: if multiple, we sum *first* candidate
            # for metrics, since HF is 1 per prompt by default.
            # If you prefer sum across all candidates, change this logic.
            if not req_out.outputs:
                completion_tokens_per_item.append(0)
                continue

            first = req_out.outputs[0]

            token_ids = getattr(first, "token_ids", None)
            if token_ids is not None:
                completion_tokens_per_item.append(len(token_ids))
            else:
                # fallback: tokenize generated text (less exact due to cleanup rules)
                completion_tokens_per_item.append(len(self.tokenizer.encode(first.text, add_special_tokens=False)))

        metrics["prompt_tokens"] = int(sum(prompt_tokens_per_item))
        metrics["completion_tokens"] = int(sum(completion_tokens_per_item))
        metrics["total_tokens"] = int(metrics["prompt_tokens"] + metrics["completion_tokens"])
        metrics["generation(s)"] = float(time.perf_counter() - t0)

        # ---- format outputs ----
        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [
                [c.text for c in out.outputs] if len(out.outputs) > 1 else out.outputs[0].text
                for out in outputs
            ]
            base_output = generated_texts

        if return_dict:
            # Optional: expose per-item counts too, similar to your HF internals
            return {
                "responses": base_output,
                "prompt_tokens_per_item": [int(x) for x in prompt_tokens_per_item],
                "completion_tokens_per_item": [int(x) for x in completion_tokens_per_item],
                "raw_outputs": outputs if return_raw_output else None,
            }

        if return_scores:
            scores = []
            for out in outputs:
                out_scores = []
                for single in out.outputs:
                    if single.logprobs:
                        # single.logprobs: list[dict[token_id -> Logprob]]
                        token_probs = [np.exp(list(score_dict.values())[0].logprob) for score_dict in single.logprobs]
                        out_scores.append(token_probs)
                    else:
                        out_scores.append([])
                scores.append(out_scores[0] if len(out_scores) == 1 else out_scores)
            return base_output, scores

        return base_output