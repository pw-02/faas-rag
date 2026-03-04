import os
import json
import csv
from typing import Any, Dict, List, Optional
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

    def evaluate(self, batch:List[Item]):
        """Calculate all metric indicators and summarize them (batch mode)."""
        for metric in self.metrics:
            try:
                metric_result, metric_scores = self.metric_class[metric].calculate_metric(batch)
                for metric_score, item in zip(metric_scores, batch):
                    item.update_evaluation_score(metric, metric_score)
            except Exception as e:
                raise RuntimeError(f"Error calculating metric {metric}: {e}")
        return batch