#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Callable, Optional, Union

import hydra
from tqdm import tqdm

from pwrag.args.args import AppConfig
from pwrag.client.run_logger import RunLogger
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialPipeline


def run_eval(
    cfg: AppConfig,
    dataset: Dataset,
    pipeline: Union[LLMOnlyPipeline, SequentialPipeline, RetrievalOnlyPipeline],
    evaluator: Evaluator,
    run_logger: Optional[RunLogger] = None,
    desc: str = "Generating + Evaluating",
) -> None:
    """Run pipeline over dataset, log per-item results, and compute accuracy where applicable."""

    if run_logger is not None:
        run_logger.save_config(cfg)

    pbar = tqdm(dataset.data, desc=desc, unit="item")

    for item in pbar:
        question = item.question
        # Run pipeline (avoid repeating this logic)
        if isinstance(pipeline, RetrievalOnlyPipeline):
            retrieved_docs, perf_metrics = pipeline.run(question=question, return_metrics=True)
            # If you want these persisted in outputs, uncomment:
            # item.update_output("retrieved_docs", retrieved_docs)

        elif isinstance(pipeline, (LLMOnlyPipeline, SequentialPipeline)):
            pred, perf_metrics = pipeline.run(question=question, return_metrics=True)          
            item.update_output("pred", pred)
            acc_metrics = evaluator.evaluate_item(item)
            item.update_metrics("acc_metrics", acc_metrics)
        else:
            raise TypeError(f"Unsupported pipeline type: {type(pipeline).__name__}")

        # Always attach perf metrics
        item.update_metrics("metrics", perf_metrics)
        
        # Logging + progress reporting (if enabled)
        if run_logger is not None:
            run_logger.log_item(item.to_dict())
            run_logger.maybe_report(pbar)

    if run_logger is not None:
        run_logger.finalize()


@hydra.main(config_path="../config", config_name="local_config", version_base=None)  # local_config.yaml, config.yaml
def main(cfg: AppConfig) -> None:
    evaluator = Evaluator(cfg)
    dataset = Dataset(cfg)

    pipeline = SequentialPipeline(cfg)
    # pipeline: PipelineT = LLMOnlyPipeline(cfg)
    # pipeline: PipelineT = RetrievalOnlyPipeline(cfg)
    
    run_logger = RunLogger(
        cfg,
        pipeline_name=pipeline.pipeline_name,
        run_name=cfg.run_name,
        overwrite=True,
        report_every=10,
        log_items=True,
        flush_every=1,
    )

    run_eval(
        cfg=cfg,
        dataset=dataset,
        pipeline=pipeline,
        evaluator=evaluator,
        run_logger=run_logger,
        desc=f"Eval: {type(pipeline).__name__}",
    )


if __name__ == "__main__":
    main()