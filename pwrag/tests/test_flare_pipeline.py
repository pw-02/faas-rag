# main.py
import hydra
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.pipeline.pipeline import FLAREPipeline
from pwrag.evaluator.evaluator import Evaluator


@hydra.main(config_path="../config", config_name="dev_config", version_base=None)
def main(cfg: AppConfig):

    evaluator = Evaluator(cfg)
    dataset = Dataset(cfg)
    pipeline = FLAREPipeline(cfg)
    batch_size = 1

    num_batches = dataset.num_batches(batch_size)
    pbar = tqdm(total=num_batches, desc="FLARE Pipeline Test", unit="item")

    for bidx, batch in enumerate(dataset.iter_batches(batch_size), start=1):
        batch, batch_perf_metrics = pipeline.run_batch(batch)
        batch_acc_metrics = evaluator.evaluate(batch)
        print(f"Batch {bidx} - Perf Metrics: {batch_perf_metrics}, Acc Metrics: {batch_acc_metrics}")
        for item in batch:
            question = item.question
            golden_answers = item.golden_answers
            pred = item.output.get("pred", "No prediction available")
        print("-" * 50)

    pbar.close()

if __name__ == "__main__":
    main()