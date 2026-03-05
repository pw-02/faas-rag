#!/usr/bin/env python3
from __future__ import annotations
import hydra
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.pipeline.search_o1 import SearchO1Pipeline
import json

def find_examples(
    dataset: Dataset,
    pipeline: SearchO1Pipeline,
    desc: str = "Finding examples",
) -> None:
    """Run pipeline over dataset, log per-batch results, and compute metrics where applicable."""
    data = []
    for item in tqdm(dataset, desc=desc, unit="item"):
        if pipeline.is_multi_retrival_example(item):
            print("Found multi-retrieval example:")
            print("Id:", item.id)
            print("Question:", item.question)
            print("-" * 50)
            data.append(item.data)
    print(f"Total multi-retrieval examples found: {len(data)}")
    #save data to jsonl file
  
    with open("multi_retrieval_examples.jsonl", "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


@hydra.main(config_path="../config", config_name="dev_config", version_base=None)  # dev_config.yaml, config.yaml
def main(cfg: AppConfig) -> None:
    if cfg.retriever.pipeline.name == "search_o1":
        print("Running evaluation with SearchO1Pipeline...")
        pipelines = [SearchO1Pipeline(cfg)]
    else:
        raise ValueError(f"Unknown pipeline name: {cfg.retriever.pipeline.name}")
    
    for pipeline in pipelines:
        dataset = Dataset(cfg)
        # pipeline = pipeline(cfg)
        find_examples(dataset, pipeline)
    
if __name__ == "__main__":
    main()