import logging
from typing import Optional
import torch
from faasrag.core.args import EmbedderConfig, IndexConfig, LlamaGeneratorConfig, ProximityCacheConfig, DocStoreConfig
from faasrag.core.embedders import build_embedder
from faasrag.core.generators import build_generator
from faasrag.core.docstores import load_docstore
from faasrag.core.indexes import load_index

class RagPipeline:
    def __init__(
        self,
        generator_cfg: LlamaGeneratorConfig,
        embedder_cfg: EmbedderConfig,
        index_cfg: IndexConfig,
        docstore_cfg: DocStoreConfig,
        artifact_dir: str,
        device: Optional[str] = None,
        cache_cfg: Optional[ProximityCacheConfig] = None,
        top_k: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = int(top_k)
        self.logger = logger or logging.getLogger("rag_pipeline")

        # 1) Embedder
        self.embedder = build_embedder(embedder_cfg, device=self.device)

        # 2) Index (fail fast)
        self.index = load_index(index_cfg, artifact_dir=artifact_dir)  # or faiss.read_index(vector_index_path)

        # # 3) Sanity check dims BEFORE loading big docstore / generator
        # self.sanity_check_dimensions()
        
        # 4) Docstore
        self.docstore = load_docstore(docstore_cfg, artifact_dir=artifact_dir)

        pass

        

    def sanity_check_dimensions(self) -> None:
        test = self.embedder.embed_queries(["dim check"])
        if test.shape[1] != self.index.d:
            raise ValueError(f"Embed dim {test.shape[1]} != index dim {self.index.d}")
        else:
            self.logger.info(f"Embedder dim {test.shape[1]} matches index dim {self.index.d}")
    


# self.cache = build_cache(cache_cfg) if cache_cfg else None

#         # Load index + docstore once
#         self.index = load_index(index_cfg, artifact_dir=artifact_dir)
#         self.docstore = load_docstore(docstore_path)

#         # Dimension check
#         test = self.embedder.embed_queries(["dim check"])
#         if test.shape[1] != self.index.d:
#             raise ValueError(f"Embed dim {test.shape[1]} != index dim {self.index.d}")

#     def run(self, query: str) -> str:
#         q = self.embedder.embed_queries([query]).astype("float32")
#         scores, ids = self.index.search(q, self.top_k)

#         # fetch docs
#         docs = []
#         for rank, idx in enumerate(ids[0]):
#             item = self.docstore.get(int(idx))
#             text = item["text"] if isinstance(item, dict) else str(item)
#             docs.append((text, float(scores[0][rank])))

#         # build prompt (simple)
#         context = "\n\n".join([f"[{i+1}] {t}" for i, (t, _) in enumerate(docs)])
#         prompt = f"Use the context to answer.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"

#         return self.generator.generate(prompt)