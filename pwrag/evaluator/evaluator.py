import os
import json
import csv
from typing import Any, Dict, Optional
from pwrag.args.args import AppConfig
from pwrag.evaluator.metrics import BaseMetric
from pwrag.dataset.dataset import Item


class Evaluator:
    """Evaluator is used to summarize the results of all metrics."""

    def __init__(self, config: AppConfig):
        self.config: AppConfig = config
        # always use normalized metric list
        self.metrics = [metric.lower() for metric in self.config.metrics]
        self.available_metrics = self._collect_metrics()
        self.metric_class: Dict[str, BaseMetric] = {}
        
        for metric in self.metrics:
            if metric in self.available_metrics:
                self.metric_class[metric] = self.available_metrics[metric](self.config)
            else:
                raise NotImplementedError(f"{metric} has not been implemented!")
      
    def _collect_metrics(self) -> Dict[str, type[BaseMetric]]:
        """Collect all classes based on BaseMetric subclasses."""
        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()
            for subclass in base_class.__subclasses__():
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses

        available: Dict[str, type[BaseMetric]] = {}
        for cls in find_descendants(BaseMetric):
            metric_name = getattr(cls, "metric_name", None)
            if isinstance(metric_name, str) and metric_name:
                available[metric_name.lower()] = cls
        return available

    def evaluate_item(self, item: Item) -> Dict[str, Any]:
        """Evaluate a single data sample."""
        result_dict: Dict[str, Any] = {}
        for metric in self.metrics:
            try:
                metric_result, metric_score = self.metric_class[metric].calculate_metric_for_item(item)
                result_dict.update(metric_result)
            except Exception as e:
                # don't crash the whole run
                result_dict[metric] = None
                result_dict[f"{metric}_error"] = str(e)
        return result_dict

    def evaluate(self, data):
        """Calculate all metric indicators and summarize them (batch mode)."""
        result_dict: Dict[str, Any] = {}
        for metric in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric].calculate_metric(data)
                result_dict.update(metric_result)

                for metric_score, item in zip(metric_scores, data):
                    item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                result_dict[metric] = None
                result_dict[f"{metric}_error"] = str(e)

        return result_dict

    def save_metric_score(self, result_dict, file_name="metric_score.txt"):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for k, v in result_dict.items():
                f.write(f"{k}: {v}\n")

    def save_data(self, data, file_name="intermediate_data.json"):
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, file_name)
        data.save(save_path)