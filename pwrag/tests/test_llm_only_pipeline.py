import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from pwrag.args.args import AppConfig
from pwrag.evaluator.evaluator import Evaluator
from pwrag.dataset.dataset import Dataset
from pwrag.pipeline.pipeline import LLMOnlyPipeline



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: AppConfig) -> None:
    print("==== CONFIG ====")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================\n")

    evaluator = Evaluator(cfg)
    evaluator.start_streaming(output_name="item_results.csv",report_every=10, overwrite=True)

    dataset = Dataset(cfg)
    print(f"Dataset: {dataset.dataset_name}, Sample num: {len(dataset.data)}")

    pipeline = LLMOnlyPipeline(cfg)

    for item in tqdm(dataset.data, desc="Generating + Evaluating", unit="item"):
        pred = pipeline.run_single(item.question)
        item.update_output("pred",pred)  # ensures item.pred exists
        
        item_metrics = evaluator.evaluate_item(item)
        evaluator.log_item(item, item_metrics)
        evaluator.maybe_report()

    evaluator.finalize_streaming()

    if cfg.save_sample_metrics:
        evaluator.save_data(dataset, file_name="intermediate_data_streaming.json")


if __name__ == "__main__":
    main()