from asyncio import run
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
from pwrag.client.run_logger import RunLogger
def run_eval(
    *,
    cfg: AppConfig,
    dataset : Dataset,
    pipeline: Union[LLMOnlyPipeline, SequentialPipeline],
    evaluator: Evaluator,
    run_logger: Optional[RunLogger] = None,
    desc: str = "Generating + Evaluating",
    get_pred: Optional[Callable[[Any, Any], Any]] = None,
):
    run_logger.save_config(cfg)
    pbar = tqdm(dataset.data, desc=desc, unit="item")
    for item in pbar:
        pred, cost_metrics = pipeline.run(question=item.question, return_metrics=True)
        item.update_output("pred", pred)
        acc_metrics = evaluator.evaluate_item(item)
        item.update_metrics("acc_metrics", acc_metrics)
        item.update_metrics("cost_metrics", cost_metrics)
        run_logger.log_item(item.to_dict())
        run_logger.maybe_report(pbar)
    run_logger.finalize()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: AppConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    evaluator = Evaluator(cfg)
    dataset = Dataset(cfg)
    # pipeline = LLMOnlyPipeline(cfg)
    pipeline = SequentialPipeline(cfg)

    run_logger = RunLogger(cfg, pipeline_name=pipeline.pipeline_name, overwrite=True, report_every=10)


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