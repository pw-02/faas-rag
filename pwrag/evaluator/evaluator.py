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

       
        self.save_sample_metrics = config.save_sample_metrics
        self.save_summary_metrics = config.save_summary_metrics

        # always use normalized metric list
        self.metrics = [metric.lower() for metric in self.config.metrics]

        self.available_metrics = self._collect_metrics()

        self.metric_class: Dict[str, BaseMetric] = {}
        for metric in self.metrics:
            if metric in self.available_metrics:
                self.metric_class[metric] = self.available_metrics[metric](self.config)
            else:
                raise NotImplementedError(f"{metric} has not been implemented!")

        # streaming state (initialized in start_streaming)
        self._stream_report_every = 0
        self._stream_n = 0
        self._stream_sum: Dict[str, float] = {}
        self._stream_cnt: Dict[str, int] = {}
        self._stream_output_path = None
        self._stream_summary_name = None
        self._stream_format = "jsonl"
        self._csv_fieldnames: Optional[list[str]] = None

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

        if self.save_summary_metrics:
            self.save_metric_score(result_dict)

        if self.save_sample_metrics:
            self.save_data(data)

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

    # ---------------- Per-item logging (your "streaming") ----------------

    def start_streaming(
        self,
        output_name: str = "item_results.jsonl",
        summary_name: str = "metric_score_streaming.txt",
        report_every: int = 10,
        overwrite: bool = False,
        csv_fixed_columns: bool = False,
    ) -> None:
        """
        If output_name ends with .jsonl => JSONL (recommended)
        If output_name ends with .csv  => CSV

        csv_fixed_columns:
          - if True, writes a fixed CSV header (stable columns). Recommended if you insist on CSV.
          - if False, CSV header is taken from the first record only (later new keys may be dropped/misaligned).
        """
        os.makedirs(self.save_dir, exist_ok=True)

        self._stream_report_every = int(report_every)
        self._stream_n = 0
        self._stream_sum = {}
        self._stream_cnt = {}

        self._stream_output_path = os.path.join(self.save_dir, output_name)
        self._stream_summary_name = summary_name

        ext = os.path.splitext(self._stream_output_path)[1].lower()
        self._stream_format = "jsonl" if ext == ".jsonl" else "csv"

        if overwrite and os.path.exists(self._stream_output_path):
            os.remove(self._stream_output_path)

        # Optional: fixed CSV header
        self._csv_fieldnames = None
        if self._stream_format == "csv" and csv_fixed_columns:
            # stable schema: basic fields + one column per metric
            self._csv_fieldnames = [
                "id", "question", "pred", "golden_answers", "choices",
                *[f"metric.{m}" for m in self.metrics],
            ]

    def _flatten_for_csv(self, record: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
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
        if self._stream_output_path is None:
            raise RuntimeError("start_streaming() must be called before log_item().")

        if self._stream_format == "jsonl":
            with open(self._stream_output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return

        # CSV
        flat = self._flatten_for_csv(record)

        # choose fieldnames
        fieldnames = self._csv_fieldnames or list(flat.keys())
        write_header = not os.path.exists(self._stream_output_path)

        with open(self._stream_output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    def log_item(self, item: Item, item_metrics: Dict[str, Any], cost_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Append one record per item to JSONL/CSV AND update running aggregates.
        You can pass cost_metrics too so you record everything per item.
        """
        self._stream_n += 1

        # running aggregates (per-metric count; handles missing metrics safely)
        for k, v in item_metrics.items():
            try:
                fv = float(v)
            except Exception:
                continue
            self._stream_sum[k] = self._stream_sum.get(k, 0.0) + fv
            self._stream_cnt[k] = self._stream_cnt.get(k, 0) + 1

        pred0 = ""
        try:
            pred0 = item.pred[0] if getattr(item, "pred", None) else ""
        except Exception:
            pred0 = ""

        record = {
            "id": getattr(item, "id", None),
            "question": getattr(item, "question", None),
            "pred": pred0,
            "golden_answers": getattr(item, "golden_answers", None),
            "choices": getattr(item, "choices", None),
            "metric": item_metrics,
        }
        if cost_metrics is not None:
            record["cost"] = cost_metrics

        self._append_record(record)

    def maybe_report(self, pbar=None) -> Optional[Dict[str, float]]:
        n = self._stream_n
        if n == 0:
            return None
        if self._stream_report_every and (n % self._stream_report_every == 0):
            avg = {
                k: (self._stream_sum[k] / self._stream_cnt[k])
                for k in self._stream_sum
                if self._stream_cnt.get(k, 0) > 0
            }
            if pbar is not None:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in avg.items()})
            return avg
        return None

    def finalize_streaming(self) -> Dict[str, float]:
        n = self._stream_n
        summary = {
            k: (self._stream_sum[k] / self._stream_cnt[k])
            for k in self._stream_sum
            if self._stream_cnt.get(k, 0) > 0
        } if n else {}

        if self.save_summary_metrics and self._stream_summary_name:
            self.save_metric_score(summary, file_name=self._stream_summary_name)

        return summary