import os
import json
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

from omegaconf import OmegaConf
from pwrag.args.args import AppConfig
from pwrag.utils.utils import AverageMeter
from pwrag.dataset.dataset import Item


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class LogPaths:
    save_dir: str
    batches_csv: str
    items_jsonl: str
    summary_csv: str
    config_yaml: str


class RunLogger:
    """
    Batch logger + dataset summary.

    Writes:
      - items.jsonl (optional): one JSON record per item
      - batches.csv (optional): one row per batch
      - summary.csv: one row per run
      - config.yaml: resolved config dump
    """

    def __init__(
        self,
        conf: AppConfig,
        pipeline_name: str = "",
        overwrite: bool = True,
        log_items: bool = False,
        log_batches: bool = True,
        # Optional: stable schema for batch CSV. If None, infer from first batch and freeze.
        batch_fieldnames: Optional[List[str]] = None,
    ) -> None:
        self.cfg = conf
        self.pipeline_name = pipeline_name
        self.dataset_name = self.cfg.dataset.dataset_name
        self.index_name = self.cfg.retriever.index.name

        self.log_items = bool(log_items)
        self.log_batches = bool(log_batches)

        self.num_batches = 0
        self.num_items = 0

        self.save_dir = self.cfg.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.paths = LogPaths(
            save_dir=self.save_dir,
            items_jsonl=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}_items.jsonl"),
            batches_csv=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}_batches.csv"),
            summary_csv=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}_summary.csv"),
            config_yaml=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}_config.yaml"),
        )

        if overwrite:
            for p in [self.paths.batches_csv, self.paths.items_jsonl, self.paths.config_yaml]:
                if os.path.exists(p):
                    os.remove(p)

        # meters across whole run
        self.metric_perf: Dict[str, AverageMeter] = {}
        self.metric_score: Dict[str, AverageMeter] = {}

        # batch CSV schema handling
        self._batch_fieldnames = batch_fieldnames
        self._batch_header_written = False

    # -------- context manager --------
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # no persistent handles; nothing special to close
        return None

    # -------- internals --------
    def _get_meter(self, pool: Dict[str, AverageMeter], key: str) -> AverageMeter:
        if key not in pool:
            pool[key] = AverageMeter(key)
        return pool[key]

    def save_config(self, conf: AppConfig) -> str:
        OmegaConf.save(config=conf, f=self.paths.config_yaml, resolve=True)
        return self.paths.config_yaml

    def conf_to_dict(self) -> Dict[str, Any]:
        return OmegaConf.to_container(self.cfg, resolve=True)

    def _append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _append_csv_row(self, path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # -------- logging --------
    def log_batch(self, batch_id: int, batch: List[Item]) -> None:
        if not batch:
            return

        self.num_batches += 1

        batch_metric_perf: Dict[str, AverageMeter] = {}
        batch_metric_score: Dict[str, AverageMeter] = {}

        for item in batch:
            self.num_items += 1
            metric_perf = item.output.get("metric_perf", {}) or {}
            metric_score = item.output.get("metric_score", {}) or {}

            for k, v in metric_score.items():
                self._get_meter(self.metric_score, k).update(v)
                self._get_meter(batch_metric_score, k).update(v)

            for k, v in metric_perf.items():
                self._get_meter(self.metric_perf, k).update(v)
                self._get_meter(batch_metric_perf, k).update(v)

            if self.log_items:
                self._append_jsonl(self.paths.items_jsonl, item.to_dict())

        if not self.log_batches:
            return

        # batch record
        record: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "dataset_name": self.dataset_name,
            "index_name": self.index_name,
            "pipeline_name": self.pipeline_name,
            "batch_id": int(batch_id),
            "batch_size": int(len(batch)),
        }

        # scores: average across items in batch
        for k, m in batch_metric_score.items():
            record[f"score.{k}"] = m.avg

        # perf: log both avg and sum so you can interpret easily
        for k, m in batch_metric_perf.items():
            # record[f"perf.{k}_avg"] = m.avg
            record[f"perf.{k}_sum"] = m.sum

        # freeze schema on first batch if not provided
        if self._batch_fieldnames is None:
            self._batch_fieldnames = list(record.keys())

        self._append_csv_row(self.paths.batches_csv, record, self._batch_fieldnames)

    def finalize(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "dataset_name": self.dataset_name,
            "index_name": self.index_name,
            "pipeline": self.pipeline_name,
            "num_items": int(self.num_items),
            "generator": self.cfg.generator.model_name,
            "num_batches": int(self.num_batches),
            "batch_size": int(self.cfg.batch_size),
            "retrieval_topk": int(self.cfg.retrieval_topk),
        }

        for k, m in self.metric_score.items():
            summary[f"score.{k}"] = m.avg

        for k, m in self.metric_perf.items():
            # summary[f"perf.{k}_avg"] = m.avg
            summary[f"perf.{k}_sum"] = m.sum

        # keep config out of CSV unless you really want giant rows
        # Instead: store it in config_yaml, and optionally add a pointer.
        summary["config_path"] = self.paths.config_yaml

        # write summary.csv
        fieldnames = list(summary.keys())
        self._append_csv_row(self.paths.summary_csv, summary, fieldnames)

        return summary
    
    def get_live_metrics(self) -> Dict[str, float]:
        metrics = {}

        if "generation_time(s)" in self.metric_perf:
            metrics["gen_s"] = round(self.metric_perf["generation_time(s)"].avg, 3)

        # if "prompt_tokens" in self.metric_perf:
        #     metrics["prompt_tok"] = round(self.metric_perf["prompt_tokens"].avg, 1)

        # if "completion_tokens" in self.metric_perf:
        #     metrics["comp_tok"] = round(self.metric_perf["completion_tokens"].avg, 1)

        return metrics