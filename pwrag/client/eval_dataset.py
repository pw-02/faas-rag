#!/usr/bin/env python3
from __future__ import annotations

# ---- MUST be first: before importing anything that might touch torch/cuda ----
#'FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION')

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # optional
# print("backend:", os.environ.get("VLLM_ATTENTION_BACKEND"))
from typing import Optional

import hydra
from tqdm import tqdm

from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import (
    LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialRAGPipeline
)
from pwrag.pipeline.active_pipeline import FLAREPipeline, RQRAGPipeline
from pwrag.logging.run_logger import RunLogger


def run_eval(
    cfg: AppConfig,
    dataset: Dataset,
    pipeline: LLMOnlyPipeline | RetrievalOnlyPipeline | SequentialRAGPipeline | FLAREPipeline,
    evaluator: Evaluator,
    run_logger: RunLogger,
    desc: str = "Generating + Evaluating",
    batch_size: int = 1,
) -> None:
    """Run pipeline over dataset, log per-batch results, and compute metrics where applicable."""

    run_logger.save_config(cfg)
    
    num_batches = dataset.num_batches(batch_size)
    with tqdm(total=num_batches, desc=desc, unit="batch") as pbar:
        for bidx, batch in enumerate(dataset.iter_batches(batch_size), start=1):
            
            batch = pipeline.run_batch(batch)

            if not isinstance(pipeline, RetrievalOnlyPipeline):
                batch = evaluator.evaluate(batch)

            run_logger.log_batch(batch_id=bidx, batch=batch)

            pbar.set_postfix(run_logger.get_live_metrics())
            pbar.update(1)

    if run_logger is not None:
        run_logger.finalize()



@hydra.main(config_path="../config", config_name="dev_config", version_base=None)  # dev_config.yaml, config.yaml

def main(cfg: AppConfig) -> None:
    if cfg.retriever.pipeline.name == "llm_only":
        pipelines = [LLMOnlyPipeline(cfg)]
        print("Running evaluation with LLMOnlyPipeline...")
    elif cfg.retriever.pipeline.name == "retrieval_only":
        pipelines = [RetrievalOnlyPipeline(cfg)]
        print("Running evaluation with RetrievalOnlyPipeline...")
    elif cfg.retriever.pipeline.name == "sequential_rag":
        pipelines = [SequentialRAGPipeline(cfg)]
        print("Running evaluation with SequentialRAGPipeline...")
    elif cfg.retriever.pipeline.name == "flare":
        pipelines = [FLAREPipeline(cfg)]
        print("Running evaluation with FLAREPipeline...")
    elif cfg.retriever.pipeline.name == "rqrag":
        print("Running evaluation with RQRAGPipeline...")
        pipelines = [RQRAGPipeline(cfg)]
    elif cfg.retriever.pipeline.name == "all":
        print("Running evaluation with all pipelines...")
        pipelines = [LLMOnlyPipeline(cfg), RetrievalOnlyPipeline(cfg), SequentialRAGPipeline(cfg), FLAREPipeline(cfg)]
    else:
        raise ValueError(f"Unknown pipeline name: {cfg.retriever.pipeline.name}")
    
    for pipeline in pipelines:
        evaluator = Evaluator(cfg)
        dataset = Dataset(cfg)
        # pipeline = pipeline(cfg)
        cfg.save_dir = os.path.join(cfg.save_dir, pipeline.pipeline_name)  # Save under subdir for each pipeline 
        logger = RunLogger(conf=cfg, 
                           pipeline_name=pipeline.pipeline_name,
                            overwrite=True, 
                            log_items=True,
                            log_batches=True)
        
        run_eval(
            cfg=cfg,
            dataset=dataset,
            pipeline=pipeline,
            evaluator=evaluator,
            run_logger=logger,
            desc=f"Eval: {pipeline.pipeline_name}",
            batch_size=cfg.batch_size
        )

if __name__ == "__main__":
    main()