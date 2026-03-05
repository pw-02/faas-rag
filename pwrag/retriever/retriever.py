import json
import os
import time
import requests

import warnings
from typing import List, Dict, Union
import functools
from tqdm import tqdm
import faiss
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pwrag.args.args import AppConfig
from pwrag.utils.utils import get_reranker, get_cache, timed
from pwrag.retriever.utils import inspect_faiss_index, load_corpus, load_docs, convert_numpy, judge_image, judge_zh, unwrap_faiss_index
from pwrag.retriever.encoder import Encoder, STEncoder, ClipEncoder
from pwrag.retriever.caches import ProximityCache
import torch

# if get_device() == "cpu":
faiss.omp_set_num_threads(1)

def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(self, query, metrics=None, num=None, return_score=False, return_timing_metrics=False):
        results, scores, time_metrics = func(self, query=query, num=num, return_score=True, return_timing_metrics=True)
        if self.use_reranker:
            results, scores = self.reranker.rerank(query, results)
            if "batch" not in func.__name__:
                results = results[0]
                scores = scores[0]
        if return_score:
            if return_timing_metrics:
                return results, scores, time_metrics
            return results, scores
        if return_timing_metrics:
            return results, time_metrics
        else:
            return results

    return wrapper


class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config:AppConfig):
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
        self.retrieval_method = self.config.retriever.encoder.retrieval_method
        self.topk = self.config.retrieval_topk
        self.device = self.config.retriever_encoder_device if "cuda" in self.config.retriever_encoder_device and torch.cuda.is_available() else "cpu"

        self.index_path = self.config.retriever.index.index_path
        self.corpus_path = self.config.retriever.corpus.corpus_path

        self.use_cache = self.config.retriever.pipeline.use_cache and self.config.retriever.cache.type != "none"

        if self.use_cache:
            self.cache = get_cache(self.config)
        else:
            self.cache = None

        self.use_reranker = self.config.retriever.pipeline.use_reranker
        if self.use_reranker:
            self.reranker = get_reranker(self.config)
        else:
            self.reranker = None

        self.silent = self._config["silent_retrieval"] if "silent_retrieval" in self._config else True
        
    def update_additional_setting(self):
        pass

    def _save_cache(self):
        self.cache = convert_numpy(self.cache)

        def custom_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4, default=custom_serializer)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    def _batch_search(self, query, num, return_score, return_timing_metrics=False):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class BaseTextRetriever(BaseRetriever):
    """Base text retriever."""

    def __init__(self, config):
        super().__init__(config)

    @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    @rerank_manager
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _batch_search_with_rerank(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _search_with_rerank(self, *args, **kwargs):
        return self._search(*args, **kwargs)


class BM25Retriever(BaseTextRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config, corpus=None):
        super().__init__(config)
        self.load_model_corpus(corpus)

    def update_additional_setting(self):
        self.backend = self._config["bm25_backend"]

    def load_model_corpus(self, corpus):
        if self.backend == "pyserini":
            # Warning: the method based on pyserini will be deprecated
            from pyserini.search.lucene import LuceneSearcher

            self.searcher = LuceneSearcher(self.index_path)
            self.contain_doc = self._check_contain_doc()
            if not self.contain_doc:
                if corpus is None:
                    self.corpus = load_corpus(self.corpus_path)
                else:
                    self.corpus = corpus
            self.max_process_num = 8
               
        elif self.backend == "bm25s":
            import Stemmer
            import bm25s

            self.corpus = load_corpus(self.corpus_path)
            is_zh = judge_zh(self.corpus[0]["contents"])

            self.searcher = bm25s.BM25.load(self.index_path, mmap=True, load_corpus=False)
            if is_zh:
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="zh")
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)
            else:
                stemmer = Stemmer.Stemmer("english")
                self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=stemmer)
                self.tokenizer.load_stopwords(self.index_path)
                self.tokenizer.load_vocab(self.index_path)

            self.searcher.corpus = self.corpus
            self.searcher.backend = "numba"

        else:
            assert False, "Invalid bm25 backend!"

    def _check_contain_doc(self):
        r"""Check if the index contains document content"""
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, metrics: dict = None, num: int = None, return_score=False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        if self.backend == "pyserini":
            is_zh = judge_zh(query)
            if is_zh:
                self.searcher.set_language("zh")
            hits = self.searcher.search(query, num)
            if len(hits) < 1:
                if return_score:
                    return [], []
                else:
                    return []

            scores = [hit.score for hit in hits]
            if len(hits) < num:
                warnings.warn("Not enough documents retrieved!")
            else:
                hits = hits[:num]

            if self.contain_doc:
                all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
                results = [
                    {
                        "id": hit.docid, 
                        "title": content.split("\n")[0].strip('"'),
                        "text": "\n".join(content.split("\n")[1:]),
                        "contents": content,
                    }
                    for content, hit in zip(all_contents, hits)
                ]
            else:
                results = load_docs(self.corpus, [hit.docid for hit in hits])
        elif self.backend == "bm25s":
            import bm25s

            # query_tokens = self.tokenizer.tokenize([query], return_as="tuple", update_vocab=False)
            query_tokens = bm25s.tokenize([query])
            results, scores = self.searcher.retrieve(query_tokens, k=num)
            results = list(results[0])
            scores = list(scores[0])
        else:
            assert False, "Invalid bm25 backend!"

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query, num: int = None, return_score=False, return_timing_metrics=False):
        if self.backend == "pyserini":
            # TODO: modify batch method
            results = []
            scores = []
            time_metrics = {}
            for _query in query:
                item_result, item_score = self._search(_query, num, True, return_timing_metrics=return_timing_metrics)
                results.append(item_result)
                scores.append(item_score)
        elif self.backend == "bm25s":
            import bm25s

            # query_tokens = self.tokenizer.tokenize(query, return_as="tuple", update_vocab=False)
            query_tokens = bm25s.tokenize(query)
            results, scores = self.searcher.retrieve(query_tokens, k=num)
        else:
            assert False, "Invalid bm25 backend!"
        results = results.tolist() if isinstance(results, np.ndarray) else results
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)
        self.faiss_index_params = {}
        self.load_corpus(corpus)
        self.load_index()
        self.load_model()


    def load_corpus(self, corpus):
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus

    def load_index(self):
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        
        # Use base index for capability checks / metadata
        base_index = unwrap_faiss_index(self.index)

        # Configure search params
        if isinstance(base_index, faiss.IndexIVF):
            nprobe = int(self.config.retriever.index.nprobe)
            print(f"Setting IVF nprobe={nprobe}")
            # nprobe is on IVF index (base_index)
            base_index.nprobe = nprobe

        elif hasattr(base_index, "hnsw"):
            ef = int(self.config.retriever.index.ef_search)
            print(f"Setting HNSW efSearch={ef}")
            base_index.hnsw.efSearch = ef

        else:
            print("Index does not support nprobe or efSearch configuration.")

        # Store metadata / params (inspect base for accurate type)
        self.faiss_index_params = inspect_faiss_index(base_index)

        print(f"Successfully loaded faiss index from {self.index_path}!")
        print(f"Index metadata: {self.faiss_index_params}")

 
    def update_additional_setting(self):
        self.query_max_length = self._config.retriever.encoder.query_max_length
        self.pooling_method = self._config.retriever.encoder.pooling_method
        self.retrieval_model_path = self._config.retriever.encoder.retrieval_model
        self.use_fp16 = self._config.retriever.encoder.use_fp16
        self.encoder_batch_size = self._config.retriever.encoder.batch_size

        self.instruction = None
        self.use_st = self._config.retriever.encoder.use_sentence_transformer
        self.use_faiss_gpu = self._config.retriever.index.use_faiss_gpu 

        # # self.retrieval_model_path = self._config["retrieval_model_path"]
        # self.use_faiss_gpu = self._config["faiss_gpu"]

    def load_model(self):
        if self.use_st:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                device=self.device,
                silent=self.silent,
            )
        else:
            # check pooling method
            self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                pooling_method=self.pooling_method,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                device=self.device,
            )

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(open(os.path.join(model_path, "1_Pooling/config.json")))
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(f"Pooling method {detect_pooling_method} is not implemented.", UserWarning)
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if detect_pooling_method is not None and detect_pooling_method != pooling_method:
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(self, query: str, 
                num: int = None, 
                return_score=False, 
                return_timing_metrics=False) -> List[Dict[str, str]]:
        
        time_metrics = {}
        
        # -----------------
        # Encode (normalize)
        # -----------------

        with timed(time_metrics, "encode_(s)"):
            query_emb = self.encoder.encode(query)
            
        # Search
        with timed(time_metrics, "total_search(s)"):
            idxs = None
            scores = None

            #check cache
            if self.use_cache and self.cache is not None:
                with timed(time_metrics, "cache_search(s)"):
                    cache_results = self.cache.find(query_emb)

                if cache_results is not None:
                    time_metrics["cache_hit"] = 1
                    idxs = cache_results
                    # idxs_1d = idxs_1d[:k]  # if cache stored more than requested
                    scores = [0] * len(cache_results)  # no scores available from cache
                    time_metrics["index_search(s)"] = 0.0
            else:
                time_metrics["cache_search(s)"] = 0.0
            
            # cache miss -> ANN search    
            if idxs is None:
                time_metrics["cache_hit"] = 0

                with timed(time_metrics, "index_search(s)"):
                    scores, idxs = self.index.search(query_emb, k=k)

                # Normalize shapes
                idxs = np.asarray(idxs).reshape(-1).astype(np.int64) #int type for indexing corpus
                scores = np.asarray(scores).reshape(-1) if scores is not None else None
                
                # store ONLY doc ids in cache
                if self.use_cache and self.cache is not None:
                    self.cache.insert(query_emb, idxs.tolist()) 
                
        # Load docs
        with timed(time_metrics, "load_docs(s)"):
            results = load_docs(self.corpus, idxs)

        if return_score:
            if return_timing_metrics:
                return results, scores, time_metrics
            return results, scores
        if return_timing_metrics:
            return results, time_metrics
        else:
            return results
        
    def _batch_search(self, 
                      query: List[str], 
                      num: int = None, 
                      return_timing_metrics=False,
                      return_score=False):

        time_metrics = {}

        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk

        batch_size = self.encoder_batch_size or 1
        results = []
        scores = []

        
        with timed(time_metrics, "encode_query_time(s)"):
            emb = self.encoder.encode(query, batch_size=batch_size, is_query=True)
        with timed(time_metrics, "index_search_time(s)"):
            scores, idxs = self.index.search(emb, k=num)
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = [idx for sublist in idxs for idx in sublist]

        with timed(time_metrics, "fetch_docs_time(s)"):
            results = load_docs(self.corpus, flat_idxs)
            results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]
        
        if return_score:
            if return_timing_metrics:
                return results, scores, time_metrics
            return results, scores
        elif return_timing_metrics:
            return results, time_metrics
        else:
            return results

class BingSearchRetriever(BaseRetriever):
    """Retriever based on Bing Search API for web search."""

    def __init__(self, config):
        super().__init__(config)
        
        # Bing Search API specific configuration
        self.bing_endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.subscription_key = config["bing_subscription_key"]
        self.market= "en-US"  # default market
        self.language = "en"  # default language
        self.timeout = 30  # default timeout for API calls
        self.session = requests.Session()  # Use a session for connection pooling
        self.session.headers.update({"Ocp-Apim-Subscription-Key": self.subscription_key})
    
    def _search(self, query: str, num: int) -> List[Dict[str, str]]:
        "perform a search using the Bing Web Search API with a set timeout"

        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        params = {
            "q": query,
            "mkt": self.market,
            "setLang": self.language,
            "textDecorations": True,
            "textFormat": "HTML"}
        try:
            response = requests.get(self.bing_endpoint, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()  # Raise exception if the request failed
            search_results = response.json()
            return search_results
        except requests.Timeout:
            print(f"Bing Web Search request timed out ({self.timeout} seconds) for query: {query}")
            return {}  # Or you can choose to raise an exception
        except requests.exceptions.RequestException as e:
            print(f"Error occurred during Bing Web Search request: {e}")
            return {}











# class SerperRetriever(BaseRetriever):
#     """Retriever based on Google Serper API for web search."""

#     def __init__(self, config):
#         super().__init__(config)
        
#         # Serper API specific configuration
#         self.api_key = config["serper_api_key"]
#         if not self.api_key:
#             raise ValueError("serper_api_key is required in config")
        
#         self.api_url = "https://google.serper.dev/search"
#         self.search_type = config["serper_search_type"] if config["serper_search_type"] else "search"  # search, news, images, etc.
#         self.location = config["serper_location"] if config["serper_location"] else None  # e.g., "United States"
#         self.gl = config["serper_gl"] if config["serper_gl"] else None  # Country code, e.g., "us"
#         self.hl = config["serper_hl"] if config["serper_hl"] else "en"  # Language, e.g., "en"
        
#     def _search(self, query: str, num: int) -> List[Dict[str, str]]:
#         """
#         Retrieve top-k relevant documents using Google Serper API.
        
#         Args:
#             query: Search query string
#             num: Number of results to return
#             return_score: Whether to return relevance scores
            
#         Returns:
#             List of dictionaries containing search results with keys:
#                 - contents: The snippet/description
#                 - title: Page title
#                 - text: Full text (same as contents for web search)
#                 - url: Page URL
#                 - score: Relevance score (if return_score=True)
#         """
#         headers = {
#             'X-API-KEY': self.api_key,
#             'Content-Type': 'application/json'
#         }
        
#         payload = {
#             'q': query,
#             'num': num,
#             'hl': self.hl
#         }
        
#         if self.location:
#             payload['location'] = self.location
#         if self.gl:
#             payload['gl'] = self.gl
        
#         try:
#             response = requests.post(
#                 self.api_url,
#                 headers=headers,
#                 json=payload,
#                 timeout=30
#             )
#             response.raise_for_status()
#             data = response.json()
            
#             results = []
            
#             # Parse organic results
#             organic_results = data.get('organic', [])
#             for idx, item in enumerate(organic_results[:num]):
#                 result = {
#                     'title': item.get('title', ''),
#                     'text': item.get('snippet', ''),
#                     'url': item.get('link', ''),
#                 }
                
#                 results.append(result)
            
#             return results
            
#         except requests.exceptions.RequestException as e:
#             print(f"Error calling Serper API: {e}")
#             return []
#         except Exception as e:
#             print(f"Unexpected error in _search: {e}")
#             return []
    
#     def search(self, query: str, num: int = None) -> List[Dict[str, str]]:
#         """
#         Single search wrapper for SerperRetriever.
#         """
#         if num is None:
#             num = self.topk
#         return self._search(query, num)
    
#     def _batch_search(self, query_list: List[str], num: int) -> List[List[Dict[str, str]]]:
#         """
#         Batch search for multiple queries.
        
#         Args:
#             query_list: List of query strings
#             num: Number of results per query
#             return_score: Whether to return relevance scores
            
#         Returns:
#             List of result lists, one for each query
#         """
#         results = []
#         if num is None:
#             num = self.topk
        
#         for query in query_list:
#             result = self._search(query, num)
#             results.append(result)
#             # Add a small delay to avoid rate limiting
#             time.sleep(0.1)
        
#         return results

#     def batch_search(self, query_list: List[str], num: int = None):
#         return self._batch_search(query_list, num)





















