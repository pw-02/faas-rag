import os
from typing import Any, Dict, Optional
from pwrag.args.args import AppConfig
from pwrag.dataset import dataset
from pwrag.evaluator.metrics import BaseMetric
from pwrag.dataset.dataset import Item
import os
import json
import csv

class Evaluator:
    """Evaluator is used to summarize the results of all metrics."""

    def __init__(self, config: AppConfig):
        self.config:AppConfig = config
        self.save_dir = config.save_dir + "/" + config.dataset.dataset_name + "/" + config.generator.name + "/"

        self.save_sample_metrics = config.save_sample_metrics
        self.save_summary_metrics = config.save_summary_metrics
        self.metrics = [metric.lower() for metric in self.config.metrics]

        self.avaliable_metrics = self._collect_metrics()
        self.metric_class = {}  
        for metric in self.metrics:
            if metric in self.avaliable_metrics:
                self.metric_class[metric] = self.avaliable_metrics[metric](self.config)
            else:
                print(f"{metric} has not been implemented!")
                raise NotImplementedError

    def _collect_metrics(self):
        """Collect all classes based on ```BaseMetric```."""

        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()

            direct_subclasses = base_class.__subclasses__()
            for subclass in direct_subclasses:
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses

        avaliable_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = cls.metric_name
            avaliable_metrics[metric_name] = cls
        return avaliable_metrics
    
    def evaluate_item(self, item: Item):
        """Evaluate a single data sample."""
        result_dict = {}
        for metric in self.metrics:
            try:
                metric_result, metric_score = self.metric_class[metric].calculate_metric_for_item(item)
                result_dict.update(metric_result)
                item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                print(f"Error in {metric}: {e}")
                continue
        return result_dict

    def evaluate(self, data):
        """Calculate all metric indicators and summarize them."""

        result_dict = {}
        for metric in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric].calculate_metric(data)
                result_dict.update(metric_result)

                for metric_score, item in zip(metric_scores, data):
                    item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                print(f"Error in {metric}: {e}")
                continue

        if self.save_summary_metrics:
            self.save_metric_score(result_dict)

        if self.save_sample_metrics:
            self.save_data(data)

        return result_dict

    def save_metric_score(self, result_dict, file_name="metric_score.txt"):
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

    def save_data(self, data, file_name="intermediate_data.json"):
        """Save the evaluated data, including the raw data and the score of each data
        sample on each metric."""

        save_path = os.path.join(self.save_dir, file_name)

        data.save(save_path)

    def start_streaming(
        self,
        output_name: str = "item_results.jsonl",
        summary_name: str = "metric_score_streaming.txt",
        report_every: int = 10,
        overwrite: bool = False,
    ) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        self._stream_report_every = report_every
        self._stream_n = 0
        self._stream_sum: Dict[str, float] = {}

        self._stream_output_path = os.path.join(self.save_dir, output_name)
        self._stream_summary_name = summary_name

        ext = os.path.splitext(self._stream_output_path)[1].lower()
        self._stream_format = "jsonl" if ext == ".jsonl" else "csv"

        if overwrite and os.path.exists(self._stream_output_path):
            os.remove(self._stream_output_path)

    def _flatten_for_csv(self, record: Dict[str, Any]) -> Dict[str, Any]:
        flat = {}
        for k, v in record.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            elif isinstance(v, (list, tuple)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        return flat

    def _append_record(self, record: Dict[str, Any]) -> None:
        if self._stream_format == "jsonl":
            with open(self._stream_output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            flat = self._flatten_for_csv(record)
            write_header = not os.path.exists(self._stream_output_path)
            with open(self._stream_output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(flat)

    def log_item(self, item, item_metrics: Dict[str, Any]) -> None:
        self._stream_n += 1

        for k, v in item_metrics.items():
            try:
                self._stream_sum[k] = self._stream_sum.get(k, 0.0) + float(v)
            except Exception:
                continue

        record = {
            "id": getattr(item, "id", None),
            "question": getattr(item, "question", None),
            "pred": (item.pred[0] if getattr(item, "pred", None) else ""),
            "golden_answers": getattr(item, "golden_answers", None),
            "choices": getattr(item, "choices", None),
            "metrics": item_metrics,
        }

        self._append_record(record)

    def maybe_report(self, pbar=None) -> Optional[Dict[str, float]]:
        n = self._stream_n
        if n == 0:
            return None

        if self._stream_report_every and (n % self._stream_report_every == 0):
            avg = {k: self._stream_sum[k] / n for k in self._stream_sum}

            # show inside tqdm bar instead of printing
            if pbar is not None:
                # keep it compact
                postfix = {k: f"{v:.3f}" for k, v in avg.items()}
                pbar.set_postfix(postfix)

            return avg

        return None

    def finalize_streaming(self, dataset) -> Dict[str, float]:
        n = self._stream_n
        summary = {k: (self._stream_sum[k] / n) for k in self._stream_sum} if n else {}

        if self.save_summary_metrics:
            self.save_metric_score(summary, file_name=self._stream_summary_name)
        
        # if self.save_sample_metrics: 
        #     self.save_data(dataset, file_name="intermediate_data_streaming.json")

        return summary