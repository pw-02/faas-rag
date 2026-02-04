from __future__ import annotations
from typing import Optional
import torch
from faasrag.core.args import EmbedderConfig, LlamaGeneratorConfig, ProximityCacheConfig
from faasrag.core.docstores import load_docstore
from faasrag.core.build_embedder import build_embedder
from faasrag.core.build_generator import build_generator
import faiss

class RagPipeline:
    def __init__(
        self,
        generator_cfg: LlamaGeneratorConfig,
        embedder_cfg: EmbedderConfig,
        vector_index_path: str,
        docstore_path: str,
        device: Optional[str] = None,
        cache_cfg: Optional[ProximityCacheConfig] = None,
        top_k: int = 5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = int(top_k)

        # Build runtime components from configs
        self.embedder = build_embedder(embedder_cfg, device=self.device)
        self.generator = build_generator(generator_cfg, device=self.device)
        self.cache = build_cache(cache_cfg) if cache_cfg else None

        # Load index + docstore once
        self.index = faiss.read_index(vector_index_path)
        self.docstore = load_docstore(docstore_path)

        # Dimension check
        test = self.embedder.embed_queries(["dim check"])
        if test.shape[1] != self.index.d:
            raise ValueError(f"Embed dim {test.shape[1]} != index dim {self.index.d}")

    def run(self, query: str) -> str:
        q = self.embedder.embed_queries([query]).astype("float32")
        scores, ids = self.index.search(q, self.top_k)

        # fetch docs
        docs = []
        for rank, idx in enumerate(ids[0]):
            item = self.docstore.get(int(idx))
            text = item["text"] if isinstance(item, dict) else str(item)
            docs.append((text, float(scores[0][rank])))

        # build prompt (simple)
        context = "\n\n".join([f"[{i+1}] {t}" for i, (t, _) in enumerate(docs)])
        prompt = f"Use the context to answer.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"

        return self.generator.generate(prompt)
