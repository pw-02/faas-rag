#!/usr/bin/env python3
from __future__ import annotations
import os
import json
from typing import Any, Callable, Optional, Union
import hydra
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialPipeline

def run_eval(
    cfg: AppConfig,
    dataset: Dataset,
    pipeline: Union[LLMOnlyPipeline, SequentialPipeline, RetrievalOnlyPipeline],
    evaluator: Evaluator,
    outout_jsonl_file: Optional[str] = "eval_results_batch.jsonl",
) -> None:
    """Run pipeline over dataset, log per-item results, and compute accuracy where applicable."""

    eval_results = {}
    

    if isinstance(pipeline, (LLMOnlyPipeline, SequentialPipeline)):
        print("Running evaluation with LLMOnlyPipeline...")
        dataset, perf_metrics = pipeline.run_dataset(dataset)
        eval_results = evaluator.evaluate(dataset)

    elif isinstance(pipeline, RetrievalOnlyPipeline):
        print("Running evaluation with RetrievalOnlyPipeline...")
        dataset, perf_metrics = pipeline.run_dataset(dataset)

    #append results to jsonl file

    with open(outout_jsonl_file, "a", encoding="utf-8") as f:
            record = {
                "pipeline": pipeline.pipeline_name,
                "dataset": dataset.dataset_path,
                "n_samples": len(dataset),
                "eval_results": eval_results,
                "perf_metrics": perf_metrics,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")




@hydra.main(config_path="../config", config_name="local_config", version_base=None)  # local_config.yaml, config.yaml
def main(cfg: AppConfig) -> None:
    outfile = "eval_results_batch.jsonl"
    
    # delete existing file if exists

    # if os.path.exists(outfile):
    #     os.remove(outfile)

    for pipeline in [LLMOnlyPipeline, SequentialPipeline, RetrievalOnlyPipeline]:
        print(f"Running evaluation with {pipeline.__name__}...")
    
        evaluator = Evaluator(cfg)
        dataset = Dataset(cfg)
        pipeline = pipeline(cfg)     
        run_eval(
            cfg=cfg,
            dataset=dataset,
            pipeline=pipeline,
            evaluator=evaluator,
        )

if __name__ == "__main__":
    main()