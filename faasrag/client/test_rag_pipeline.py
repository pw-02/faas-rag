import logging
import hydra
from faasrag.core.args import RagServiceConfig
from faasrag.core.prompt_rag_pipeline import RagPipeline

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: RagServiceConfig):
    logging.basicConfig(level=logging.INFO)

    print("Loaded config")


    pipeline = RagPipeline(
        generator_cfg=cfg.generator,
        embedder_cfg=cfg.embedder,
        index_cfg=cfg.index,
        docstore_cfg=cfg.docstore,
        docstore_backend=cfg.docstore_backend,
        artifact_dir=cfg.artifact_dir,
        prompt_build_method=cfg.prompt_build_method,
        max_ctx_chars=cfg.max_ctx_chars,
        cache_cfg=cfg.cache if hasattr(cfg, "cache") else None,
        top_k=cfg.top_k,
        retrieve_only=cfg.retrieve_only,
        seed=cfg.seed,
    )

    print("Pipeline initialized")

    query = "Who wrote The Hobbit?"

    print("Running query:", query)

    out = pipeline.run(query)

    print("\n=== RESULT ===")
    for k, v in out.items():
        print(f"{k}: {v}")



if __name__ == "__main__":
    main()