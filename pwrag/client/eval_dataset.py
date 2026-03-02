#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Optional
import hydra
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialRAGPipeline, FLAREPipeline
from pwrag.logging.run_logger import RunLogger  # <-- update import path

def run_eval(
    cfg: AppConfig,
    dataset: Dataset,
    pipeline: Optional[LLMOnlyPipeline | RetrievalOnlyPipeline | SequentialRAGPipeline | FLAREPipeline],
    evaluator: Evaluator,
    run_logger: Optional[RunLogger] = None,
    desc: str = "Generating + Evaluating",
    batch_size: int = 1
) -> None:
    """Run pipeline over dataset, log per-item results, and compute accuracy where applicable."""

    if run_logger is not None:
        run_logger.save_config(cfg)

    num_batches = dataset.num_batches(batch_size)
    pbar = tqdm(total=num_batches, desc=desc, unit="item")

    # pbar = tqdm(dataset.data, desc=desc, unit="item")

    for bidx, batch in enumerate(dataset.iter_batches(batch_size), start=1):
        batch, batch_perf_metrics = pipeline.run_batch(batch)
        if not isinstance(pipeline, RetrievalOnlyPipeline):
            batch_acc_metrics = evaluator.evaluate(batch)
        else:
            batch_acc_metrics = {}       
        
        run_logger.log_batch(
            batch_id=bidx,
            items=batch.data,  # Dataset holds items in .data
            batch_perf_metrics=batch_perf_metrics,
            batch_acc_metrics=batch_acc_metrics,
        )
        pbar.update(len(batch))
        run_logger.maybe_report(pbar)
    
    run_logger.finalize()
    pbar.close()


@hydra.main(config_path="../config", config_name="dev_config", version_base=None)  # dev_config.yaml, config.yaml

def main(cfg: AppConfig) -> None:
    if cfg.retriever.pipeline.name == "llm_only":
        pipelines = [LLMOnlyPipeline]
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
    elif cfg.retriever.pipeline.name == "all":
        print("Running evaluation with all pipelines...")
        pipelines = [LLMOnlyPipeline, RetrievalOnlyPipeline(cfg), SequentialRAGPipeline(cfg), FLAREPipeline(cfg)]
    else:
        raise ValueError(f"Unknown pipeline name: {cfg.retriever.pipeline.name}")
    
    for pipeline in pipelines:
        evaluator = Evaluator(cfg)
        dataset = Dataset(cfg)
        pipeline = pipeline(cfg)
        cfg.save_dir = os.path.join(cfg.save_dir, pipeline.pipeline_name)  # Save under subdir for each pipeline 
        logger = RunLogger(conf=cfg, pipeline_name=pipeline.pipeline_name,log_batches=False)
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