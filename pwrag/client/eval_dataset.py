from typing import Any, Callable, Optional, Union
from tqdm import tqdm
# main.py
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.evaluator.evaluator import Evaluator
from pwrag.dataset.dataset import Dataset
from pwrag.pipeline.pipeline import LLMOnlyPipeline, SequentialPipeline

def run_eval(
    *,
    cfg: AppConfig,
    dataset : Dataset,
    pipeline: Union[LLMOnlyPipeline, SequentialPipeline],
    evaluator: Evaluator,
    output_name: str = "item_results.jsonl",
    report_every: int = 10,
    overwrite: bool = True,
    desc: str = "Generating + Evaluating",
    get_pred: Optional[Callable[[Any, Any], Any]] = None,
):
    """
    Generic streaming evaluation loop.

    - dataset: object with .data iterable of Items
    - pipeline: any object
    - get_pred: optional function (pipeline, item) -> pred
               if None, tries pipeline.run_item(item) then pipeline.run_single(item.question)
    """
    evaluator.start_streaming(output_name=output_name,report_every=report_every,overwrite=overwrite,)

    pbar = tqdm(dataset.data, desc=desc, unit="item")
    for item in pbar:
        pred = pipeline.run(item.question)
        item.update_output("pred", pred)
        item_metrics = evaluator.evaluate_item(item)
        evaluator.log_item(item, item_metrics)
        evaluator.maybe_report(pbar)
    
    summary = evaluator.finalize_streaming(dataset=dataset)
    return summary



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: AppConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    evaluator = Evaluator(cfg)
    dataset = Dataset(cfg)
    # llm_only_pipeline = LLMOnlyPipeline(cfg)
    pipeline = SequentialPipeline(cfg)

    run_eval(
        cfg=cfg,
        dataset=dataset,
        pipeline=pipeline,
        evaluator=evaluator,
        output_name="item_results.jsonl",
        report_every=10,
        overwrite=True,
        desc=f"Eval: {type(pipeline).__name__}",
    )


if __name__ == "__main__":
    main()