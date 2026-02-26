#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional
import hydra
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialPipeline
from pwrag.logging.run_logger import RunLogger  # <-- update import path

def run_eval(
    cfg: AppConfig,
    dataset: Dataset,
    pipeline: SequentialPipeline,
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


@hydra.main(config_path="../config", config_name="local_config", version_base=None)  # local_config.yaml, config.yaml
def main(cfg: AppConfig) -> None:

    for pipeline in [SequentialPipeline]:
        print(f"Running evaluation with {pipeline.__name__}...")
        evaluator = Evaluator(cfg)
        dataset = Dataset(cfg)
        pipeline = pipeline(cfg)     
        logger = RunLogger(
            conf=cfg,
            pipeline_name=pipeline.pipeline_name,
            report_every_items=10,               # tqdm postfix refresh cadence
            weight_batch_metrics_by_size=True,  # whether to weight batch metrics by batch size when updating item-average meters
            flush_every=1)
        
        run_eval(
            cfg=cfg,
            dataset=dataset,
            pipeline=pipeline,
            evaluator=evaluator,
            run_logger=logger,
            desc=f"Eval: {type(pipeline).__name__}",
            batch_size=cfg.batch_size
        )

if __name__ == "__main__":
    main()
