from typing import Any, Dict, List, Optional
from copy import deepcopy
from omegaconf import OmegaConf
import torch
from tqdm.auto import trange
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from pwrag.args.args import AppConfig
from pwrag.generator.utils import resolve_max_tokens
import requests

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
        generation_params["max_new_tokens"] = 28000  # set a default max_tokens to avoid vLLM error; will be resolved properly in resolve_max_tokens

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
                max_length=self.max_input_len,
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
    


class OpenAIAPIGenerator(BaseGenerator):
    """
    Generator that calls an OpenAI-compatible API (e.g., vLLM openai.api_server)
    instead of running inference locally.

    Expected endpoints:
      - /v1/completions
      - /v1/chat/completions
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)

        # Tokenizer is optional but useful for fallback token counting.
        # If you don't want to download tokenizer on client machines, you can skip this.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        except Exception:
            self.tokenizer = None

        self.session = requests.Session()

    def update_additional_setting(self):
        # Example config keys (optional):
        #   generator_api_base: "http://134.197.95.82:8000"
        #   generator_api_key: ""   (usually empty for self-hosted vLLM)
        #   generator_api_mode: "completion" | "chat"
        #   generator_api_timeout: 120

        self.api_base = self._config.generator.openai_endpoint
        self.api_key = ""
        self.api_mode = "chat" # "completion" or "chat"
        self.timeout = 120

        # vLLM OpenAI server expects a "model" field in requests.
        # Often it can be anything, but best is to use model_path or model_name.
        self.api_model = self._config.generator.model_name

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _map_params(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(generation_params)

        if "do_sample" in p:
            do_sample = p.pop("do_sample")
            if not do_sample:
                p["temperature"] = 0

        if "max_new_tokens" in p and "max_tokens" not in p:
            p["max_tokens"] = p.pop("max_new_tokens")

        allowed = {
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "seed",
            "logprobs",
            "n",
            "presence_penalty",
            "frequency_penalty",
        }

        out = {k: v for k, v in p.items() if k in allowed}

        if "top_k" in p:
            out["top_k"] = p["top_k"]
        if "repetition_penalty" in p:
            out["repetition_penalty"] = p["repetition_penalty"]

        return out

    def _count_prompt_tokens_fallback(self, prompts: List[str]) -> List[int]:
        if self.tokenizer is None:
            return [0] * len(prompts)
        tok = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_len
        )
        if "attention_mask" in tok:
            return [int(x) for x in tok["attention_mask"].sum(dim=1).tolist()]
        pad_id = self.tokenizer.pad_token_id
        return [int(x) for x in (tok["input_ids"] != pad_id).sum(dim=1).tolist()]

    # def generate(
    #     self,
    #     input_list: List[str],
    #     return_raw_output: bool = False,
    #     return_scores: bool = False,
    #     return_token_counts: bool = False,
    #     **params,
    # ):
    #     if isinstance(input_list, str):
    #         input_list = [input_list]

    #     generation_params = deepcopy(self.generation_params)
    #     generation_params.update(params)

    #     # keep your existing token-limit conflict handling consistent
    #     generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)
    #     generation_params = self._map_params(generation_params)

    #     # stop = generation_params.get("stop")
    #     # if stop is None:
    #     #     generation_params["stop"] = ["<|eot_id|>"]
    #     # elif isinstance(stop, str):
    #     #     generation_params["stop"] = [stop, "<|eot_id|>"]
    #     # else:
    #     #     generation_params["stop"] = list(stop) + ["<|eot_id|>"]

    #     if "stop" in generation_params:
    #         generation_params["stop"].append("<|eot_id|>")
    #         generation_params["include_stop_str_in_output"] = True
    #     else:
    #         generation_params["stop"] = ["<|eot_id|>"]

    #     # If user asked for scores, request logprobs if supported
    #     # (OpenAI-style: "logprobs": True / int depending on endpoint; vLLM often accepts int)
    #     if return_scores and "logprobs" not in generation_params:
    #         generation_params["logprobs"] = 5  # small default; increase if you need
        
    #     generation_params["skip_special_tokens"] = False

    #     # Token counting fallba        prompt_token_counts = None
    #     if return_token_counts:
    #         prompt_token_counts = self._count_prompt_tokens_fallback(input_list)

    #     outputs_text = []
    #     scores = []
    #     completion_token_counts = []

    #     # Choose endpoint
    #     use_chat = (self.api_mode.lower() == "chat")

    #     for prompt in input_list:
    #         if use_chat:
    #             url = f"{self.api_base}/v1/chat/completions"
    #             payload = {
    #                 "model": self.api_model,
    #                 "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
    #                 **generation_params,
    #             }
    #         else:
    #             url = f"{self.api_base}/v1/completions"
    #             payload = {
    #                 "model": self.api_model,
    #                 "prompt": prompt,
    #                 **generation_params,
    #             }

    #         r = self.session.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
    #         r.raise_for_status()
    #         data = r.json()

    #         if return_raw_output:
    #             outputs_text.append(data)
    #             # still try to attach token counts if requested
    #         else:
    #             if use_chat:
    #                 text = data["choices"][0]["message"]["content"]
    #             else:
    #                 text = data["choices"][0]["text"]
    #             outputs_text.append(text)

    #         # scores (best-effort)
    #         if return_scores:
    #             # vLLM may return token logprobs under choices[0]["logprobs"]
    #             lp = data["choices"][0].get("logprobs")
    #             if lp and "token_logprobs" in lp and lp["token_logprobs"] is not None:
    #                 # convert logprobs -> probs
    #                 token_probs = [float(np.exp(x)) if x is not None else None for x in lp["token_logprobs"]]
    #                 scores.append(token_probs)
    #             else:
    #                 scores.append([])

    #         # token usage if available
    #         if return_token_counts:
    #             usage = data.get("usage", {})
    #             c = usage.get("completion_tokens")
    #             if c is None:
    #                 # fallback: tokenize output text
    #                 if self.tokenizer is not None and not return_raw_output:
    #                     c = len(self.tokenizer.encode(outputs_text[-1], add_special_tokens=False))
    #                 else:
    #                     c = 0
    #             completion_token_counts.append(int(c))

    #     if return_token_counts:
    #         # If server provided prompt_tokens, prefer them, else fallback counts
    #         # (vLLM typically returns usage.prompt_tokens, but not always)
    #         server_prompt_counts = []
    #         for i, out in enumerate(outputs_text):
    #             # if return_raw_output, out is dict; else no access -> use fallback
    #             if return_raw_output and isinstance(out, dict):
    #                 p = out.get("usage", {}).get("prompt_tokens")
    #                 server_prompt_counts.append(p)
    #             else:
    #                 server_prompt_counts.append(None)

    #         final_prompt_counts = []
    #         for i in range(len(input_list)):
    #             if server_prompt_counts[i] is not None:
    #                 final_prompt_counts.append(int(server_prompt_counts[i]))
    #             else:
    #                 final_prompt_counts.append(int(prompt_token_counts[i]) if prompt_token_counts else 0)

    #         total_token_counts = [p + c for p, c in zip(final_prompt_counts, completion_token_counts)]
    #         token_info = {
    #             "prompt_token_counts": final_prompt_counts,
    #             "completion_token_counts": completion_token_counts,
    #             "total_token_counts": total_token_counts,
    #         }

    #     if return_scores:
    #         if return_token_counts:
    #             return outputs_text, scores, token_info
    #         return outputs_text, scores

    #     if return_token_counts:
    #         return outputs_text, token_info

    #     return outputs_text

    def _count_prompt_tokens_for_item(self, prompt, use_chat: bool) -> int:
        if self.tokenizer is None:
            return 0
        try:
            if use_chat:
                messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    rendered = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    return len(self.tokenizer.encode(rendered, add_special_tokens=False))
                
                text = "\n".join(m.get("content", "") for m in messages)
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))
        except Exception:
            return 0
    def generate(
        self,
        input_list: List[str],
        return_raw_output: bool = False,
        return_scores: bool = False,
        return_token_counts: bool = False,
        **params,
    ):
        if isinstance(input_list, str):
            input_list = [input_list]
        if isinstance(input_list, dict):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        generation_params = resolve_max_tokens(
            params,
            generation_params,
            prioritize_new_tokens=False,
        )
        generation_params = self._map_params(generation_params)

        stop = generation_params.get("stop")
        if stop is None:
            generation_params["stop"] = ["<|eot_id|>"]
        elif isinstance(stop, str):
            generation_params["stop"] = [stop, "<|eot_id|>"]
        else:
            generation_params["stop"] = list(stop) + ["<|eot_id|>"]

        generation_params["include_stop_str_in_output"] = True
        generation_params["skip_special_tokens"] = False

        if return_scores and "logprobs" not in generation_params:
            generation_params["logprobs"] = 5

        outputs_text = []
        raw_outputs = []
        scores = []
        use_chat = self.api_mode.lower() == "chat"

        for prompt in input_list:
            if use_chat:
                url = f"{self.api_base}/v1/chat/completions"
                payload = {
                    "model": self.api_model,
                    "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
                    **generation_params,
                }
            else:
                url = f"{self.api_base}/v1/completions"
                payload = {
                    "model": self.api_model,
                    "prompt": prompt,
                    **generation_params,
                }

            r = self.session.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            if not r.ok:
                print("STATUS:", r.status_code)
                # print("PAYLOAD:", payload)
                print("RESPONSE:", r.text)
            r.raise_for_status()
            data = r.json()
            raw_outputs.append(data)

            if return_raw_output:
                outputs_text.append(data)
            else:
                if use_chat:
                    text = data["choices"][0]["message"]["content"]
                else:
                    text = data["choices"][0]["text"]
                outputs_text.append(text)

            if return_scores:
                lp = data["choices"][0].get("logprobs")
                if lp and "token_logprobs" in lp and lp["token_logprobs"] is not None:
                    token_probs = [float(np.exp(x)) if x is not None else None for x in lp["token_logprobs"]]
                    scores.append(token_probs)
                else:
                    scores.append([])

        if return_token_counts:
            prompt_token_counts = []
            completion_token_counts = []
            total_token_counts = []

            for i, data in enumerate(raw_outputs):
                usage = data.get("usage", {})

                p = usage.get("prompt_tokens")
                c = usage.get("completion_tokens")
                t = usage.get("total_tokens")

                if p is None:
                    p = self._count_prompt_tokens_for_item(input_list[i], use_chat=use_chat)

                if c is None:
                    if self.tokenizer is not None and not return_raw_output:
                        c = len(self.tokenizer.encode(outputs_text[i], add_special_tokens=False))
                    else:
                        c = 0

                if t is None:
                    t = int(p) + int(c)

                prompt_token_counts.append(int(p))
                completion_token_counts.append(int(c))
                total_token_counts.append(int(t))

            token_info = {
                "prompt_token_counts": prompt_token_counts,
                "completion_token_counts": completion_token_counts,
                "total_token_counts": total_token_counts,
            }

        if return_scores:
            if return_token_counts:
                return outputs_text, scores, token_info
            return outputs_text, scores

        if return_token_counts:
            return outputs_text, token_info

        return outputs_text











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

