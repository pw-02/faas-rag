#!/usr/bin/env python3
from __future__ import annotations

import hydra
from tqdm import tqdm

from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.evaluator.evaluator import Evaluator
from pwrag.pipeline.pipeline import LLMOnlyPipeline, RetrievalOnlyPipeline, SequentialPipeline
from pwrag.logging.run_logger import RunLogger  # <-- update import path

BATCH_SIZE = 2  # default if not in cfg


@hydra.main(config_path="../config", config_name="local_config", version_base=None)
def main(cfg: AppConfig) -> None:
    dataset = Dataset(cfg)
    evaluator = Evaluator(cfg)
    pipeline = SequentialPipeline(cfg) 
    batch_size = getattr(cfg, "batch_size", BATCH_SIZE)

    logger = RunLogger(
        cfg,
        pipeline_name=pipeline.pipeline_name,
        dataset_name=dataset.dataset_name,
        log_batches=True,                    # batch-level JSONL
        store_item_details_in_batch=False,   # keep batch logs small
        report_every_items=50,               # tqdm postfix refresh cadence
        flush_every=1,
        weight_batch_metrics_by_size=True,     # treat batch metrics as per-item averages and weight by batch_size
        fsync=False)
    
    logger.save_config(cfg)

    pbar = tqdm(total=len(dataset), desc="Evaluating")

    for bidx, batch in enumerate(dataset.iter_batches(batch_size), start=1):
        print(f"Processing batch {bidx} with {len(batch)} items...")
        batch, batch_perf_metrics = pipeline.run_batch(batch)
        if not isinstance(pipeline, RetrievalOnlyPipeline):
            batch_acc_metrics = evaluator.evaluate(batch)
        else:
            batch_acc_metrics = {}       
        logger.log_batch(
            batch_id=bidx,
            items=batch.data,  # Dataset holds items in .data
            batch_perf_metrics=batch_perf_metrics,
            batch_acc_metrics=batch_acc_metrics,
        )
        pbar.update(len(batch))
        logger.maybe_report(pbar)

    summary = logger.finalize()
    pbar.close()

    # print("Done. Summary:")
    # print(summary)


if __name__ == "__main__":
    main()