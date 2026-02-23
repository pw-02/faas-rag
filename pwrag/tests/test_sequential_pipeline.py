# main.py
import hydra
from tqdm import tqdm
from pwrag.args.args import AppConfig
from pwrag.dataset.dataset import Dataset
from pwrag.pipeline.pipeline import SequentialPipeline
from pwrag.evaluator.evaluator import Evaluator


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: AppConfig):

    evaluator = Evaluator(cfg)
    dataset = Dataset(cfg)
    pipeline = SequentialPipeline(cfg)

    pbar = tqdm(dataset.data, desc="Sequential Pipeline Test", unit="item")
    for item in pbar:
        pred = pipeline.run(item.question)
        item.update_output("pred", pred)
        eval_result = evaluator.evaluate_item(item)
        print(f"Question: {item.question}")
        print(f"Golden Answer: {item.golden_answers}")
        print(f"Prediction: {pred}")
        print(f"Perf: {eval_result}")
        print("-" * 50)

if __name__ == "__main__":
    main()