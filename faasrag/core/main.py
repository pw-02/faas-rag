# faasrag/app.py
import hydra
from omegaconf import OmegaConf
import torch

from faasrag.core.args import RagServiceConfig
from faasrag.core.rag import RagPipeline

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):

    device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device

    rag = RagPipeline(
        generator_cfg=cfg.generator,
        embedder_cfg=cfg.embedder,
        vector_index_path=cfg.vector_index_path,
        docstore_path=cfg.docstore_path,
        top_k=cfg.top_k,
        device=device,
    )

    print(rag.run("Who wrote Pride and Prejudice?"))

if __name__ == "__main__":
    main()
