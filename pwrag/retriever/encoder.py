from typing import List, Union, Optional, Dict, Any
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from pwrag.retriever.utils import load_model, pooling, parse_query, parse_image
from pwrag.utils.utils import get_device


class Encoder:
    """
    HF/Transformers-style encoder (your existing path) that uses load_model() and pooling().
    Works for many encoder checkpoints and keeps your DPR normalization behavior:
    - DPR: do NOT normalize by default (classic DPR uses inner product)
    - Others: normalize by default
    """

    def __init__(
        self,
        model_name,
        model_path,
        pooling_method,
        max_length,
        use_fp16=True,
        instruction=None,
        silent=False,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.silent = silent
        self.gpu_num = torch.cuda.device_count()
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode()
    def single_batch_encode(self, query_list: Union[List[str], str], is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(get_device()) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__ or (
            isinstance(self.model, torch.nn.DataParallel) and "T5" in type(self.model.module).__name__
        ):
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            pooler_output = output.get("pooler_output", None)
            last_hidden_state = output.get("last_hidden_state", None)
            query_emb = pooling(pooler_output, last_hidden_state, inputs["attention_mask"], self.pooling_method)

        # DPR typically uses inner product without normalization (keep your behavior)
        if "dpr" not in self.model_name.lower():
            query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy().astype(np.float32, order="C")
        return query_emb

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: ", disable=self.silent):
            query_emb.append(self.single_batch_encode(query_list[i : i + batch_size], is_query))
        return np.concatenate(query_emb, axis=0)

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, is_query=True) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        return self.encode(query_list, batch_size, is_query)


class STEncoder:
    """
    SentenceTransformers encoder for ST-native embedding models.
    NOTE: facebook/dpr-question_encoder-single-nq-base is NOT ST-native.
    For DPR checkpoints, this class will automatically switch to Transformers DPR encoders.

    Behavior:
    - For DPR:
        uses DPRQuestionEncoder (queries) or DPRContextEncoder (passages)
        normalization default: False (classic DPR/IP). You can override normalize_embeddings.
    - For ST models:
        uses SentenceTransformer.encode(normalize_embeddings=True by default)
    """

    def __init__(self, model_name, model_path, max_length, use_fp16, instruction, device, silent=True):
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.silent = silent
        self.device = device if "cuda" in device and torch.cuda.is_available() else "cpu"

        self._is_dpr = "dpr" in (model_name or "").lower() or "dpr" in (model_path or "").lower()

        self._st_model = None
        self._dpr_models: Dict[str, Any] = {}
        self._dpr_tokenizers: Dict[str, Any] = {}

        if not self._is_dpr:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                model_path,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16 if use_fp16 else torch.float},
            )

    def _device(self) -> str:
        return self.device

    def _get_dpr_pair(self, is_query: bool):
        """
        Lazily load DPR question/context encoder depending on is_query.
        """
        from transformers import (
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizer,
            DPRContextEncoder,
            DPRContextEncoderTokenizer,
        )

        key = "question" if is_query else "context"
        if key in self._dpr_models:
            return self._dpr_models[key], self._dpr_tokenizers[key]

        device = self._device()

        if is_query:
            tok = DPRQuestionEncoderTokenizer.from_pretrained(self.model_path)
            mdl = DPRQuestionEncoder.from_pretrained(self.model_path).to(device)
        else:
            tok = DPRContextEncoderTokenizer.from_pretrained(self.model_path)
            mdl = DPRContextEncoder.from_pretrained(self.model_path).to(device)

        mdl.eval()
        if self.use_fp16 and device == "cuda":
            mdl = mdl.half()

        self._dpr_models[key] = mdl
        self._dpr_tokenizers[key] = tok
        return mdl, tok

    @torch.inference_mode()
    def _encode_dpr(
        self,
        query_list: Union[List[str], str],
        batch_size: int = 64,
        is_query: bool = True,
        normalize_embeddings: Optional[bool] = None,
    ) -> np.ndarray:
        import torch.nn.functional as F

        # DPR: parse_query still fine (it should return strings; DPR should not add instructions)
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)
        if isinstance(query_list, str):
            query_list = [query_list]

        mdl, tok = self._get_dpr_pair(is_query=is_query)
        device = self._device()

        # DPR default: do NOT normalize unless user explicitly asks
        if normalize_embeddings is None:
            normalize_embeddings = False

        outs = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: ", disable=self.silent):
            batch = query_list[i : i + batch_size]
            inputs = tok(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            # DPR*Encoder returns pooler_output for embeddings
            emb = mdl(**inputs).pooler_output  # [B, 768]

            if normalize_embeddings:
                emb = F.normalize(emb, p=2, dim=1)

            outs.append(emb.float().cpu())

        arr = torch.cat(outs, dim=0).numpy().astype(np.float32, order="C")
        return arr

    @torch.inference_mode()
    def encode(
        self,
        query_list: Union[List[str], str],
        batch_size: int = 64,
        is_query: bool = True,
        normalize_embeddings: Optional[bool] = None,
    ) -> np.ndarray:
        """
        If DPR checkpoint -> use Transformers DPR encoders.
        Otherwise -> use SentenceTransformer.encode().

        normalize_embeddings:
          - ST models: default True (keeps your previous behavior)
          - DPR: default False (classic DPR/IP). You can set True if your index uses cosine-like behavior.
        """
        if self._is_dpr:
            return self._encode_dpr(
                query_list=query_list,
                batch_size=batch_size,
                is_query=is_query,
                normalize_embeddings=normalize_embeddings,
            )

        # ST-native path
        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)

        # ST default: normalize unless user overrides
        if normalize_embeddings is None:
            normalize_embeddings = True

        query_emb = self._st_model.encode(
            query_list,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=not self.silent,
        )
        return query_emb.astype(np.float32, order="C")

    @torch.inference_mode()
    def multi_gpu_encode(
        self,
        query_list: Union[List[str], str],
        batch_size: Optional[int] = None,
        is_query: bool = True,
        normalize_embeddings: Optional[bool] = None,
    ) -> np.ndarray:
        # DPR multi-process pool doesn't apply; DPR uses standard torch batching above.
        if self._is_dpr:
            return self._encode_dpr(
                query_list=query_list,
                batch_size=batch_size or 64,
                is_query=is_query,
                normalize_embeddings=normalize_embeddings,
            )

        query_list = parse_query(self.model_name, query_list, self.instruction, is_query)

        if normalize_embeddings is None:
            normalize_embeddings = True

        pool = self._st_model.start_multi_process_pool()
        try:
            query_emb = self._st_model.encode_multi_process(
                query_list,
                pool,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=not self.silent,
            )
        finally:
            self._st_model.stop_multi_process_pool(pool)

        return query_emb.astype(np.float32, order="C")


class ClipEncoder:
    """ClipEncoder class for encoding queries using CLIP."""

    def __init__(self, model_name, model_path, silent=False):
        self.model_name = model_name
        self.model_path = model_path
        self.silent = silent
        self.gpu_num = torch.cuda.device_count()
        self.load_clip_model()

    def load_clip_model(self):
        from transformers import AutoModel, AutoProcessor

        with open(os.path.join(self.model_path, "config.json")) as f:
            config = json.load(f)
        model_type = config.get("architectures", [None])[0]
        self.model_type = model_type

        if model_type == "CLIPModel" or model_type == "ChineseCLIPModel":
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        elif model_type and model_type.endswith("CLIPModel"):
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.processor = None
        else:
            raise NotImplementedError(f"Unsupported model type: {model_type}")

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # set model max length for model that not specified in config.json
        if self.processor is not None and getattr(self.processor, "tokenizer", None) is not None:
            if self.processor.tokenizer.model_max_length > 100000:
                try:
                    model_max_length = config["text_config"]["max_position_embeddings"]
                except Exception:
                    model_max_length = 512
                self.processor.tokenizer.model_max_length = model_max_length

    @torch.inference_mode()
    def single_batch_encode(self, query_list: Union[List[str], str], modal="image") -> np.ndarray:
        encode_func_dict = {"text": self.encode_text, "image": self.encode_image}
        return encode_func_dict[modal](query_list)

    @torch.inference_mode()
    def encode(self, query_list: Union[List[str], str], batch_size=64, modal="image") -> np.ndarray:
        if not isinstance(query_list, list):
            query_list = [query_list]
        query_emb = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: ", disable=self.silent):
            query_emb.append(self.single_batch_encode(query_list[i : i + batch_size], modal))
        return np.concatenate(query_emb, axis=0)

    @torch.inference_mode()
    def multi_gpu_encode(self, query_list: Union[List[str], str], batch_size=64, modal="image") -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        return self.encode(query_list, batch_size, modal)

    @torch.inference_mode()
    def encode_image(self, image_list: List) -> np.ndarray:
        # Each item in image_list: PIL Image, local path, or URL
        if self.model_type == "CLIPModel" or self.model_type == "ChineseCLIPModel":
            image_list = [parse_image(image) for image in image_list]
            inputs = self.processor(images=image_list, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            image_emb = self.model.get_image_features(**inputs)
            image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            image_emb = image_emb.detach().cpu().numpy().astype(np.float32)
        elif self.model_type.endswith("CLIPModel"):
            image_emb = self.model.encode_image(image_list)
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_type}")
        return image_emb

    @torch.inference_mode()
    def encode_text(self, text_list: List[str]) -> np.ndarray:
        if self.model_type == "CLIPModel" or self.model_type == "ChineseCLIPModel":
            inputs = self.processor(text=text_list, padding=True, truncation=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            text_emb = self.model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
            text_emb = text_emb.detach().cpu().numpy().astype(np.float32)
        elif self.model_type.endswith("CLIPModel"):
            text_emb = self.model.encode_text(text_list, padding=True, truncation=True)
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_type}")
        return text_emb