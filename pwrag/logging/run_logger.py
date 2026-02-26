import os
import json
import csv
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
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
    batches_jsonl: str
    summary_csv: str
    config_yaml: str


class RunLogger:
    """
    Batch logger + dataset summary.

    Logs:
      - batch JSONL: one record per batch (with optional embedded items)
      - dataset CSV: includes BOTH per-item and per-batch averages

    Definitions:
      - avg_item.* : item-weighted average (each item counts equally)
      - avg_batch.*: batch average (each batch counts equally)
    """

    def __init__(
        self,
        conf: AppConfig,
        pipeline_name: str = "",
        overwrite: bool = True,
        log_batches: bool = True,
        store_item_details_in_batch: bool = False,
        flush_every: int = 10,
        fsync: bool = False,
        report_every_items: int = 10,
        report_perf_keys: Optional[List[str]] = None,
        # If your batch metrics are per-item averages (typical), keep this True.
        # It affects ONLY avg_item.* meters.
        weight_batch_metrics_by_size: bool = True,
    ) -> None:
        self.cfg = conf
        self.pipeline_name = pipeline_name
        self.dataset_name = self.cfg.dataset.dataset_name
        self.dataset_path = self.cfg.dataset.dataset_path
        self.index_path = self.cfg.retriever.index.index_path
        self.index_name = self.cfg.retriever.index.name

        self.flush_every = int(flush_every)
        self.fsync = bool(fsync)
        self.save_dir = self.cfg.save_dir
        self.log_batches = bool(log_batches)
        self.store_item_details_in_batch = bool(store_item_details_in_batch)
        self.report_every_items = int(report_every_items)
        
        self.report_perf_keys = report_perf_keys or [
            "encode_query_time(s)",
            "search_time(s)",
            "generation_time(s)",
            "cache_hit",
        ]
        self.weight_batch_metrics_by_size = bool(weight_batch_metrics_by_size)
        start = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

        self.paths = LogPaths(
            save_dir=self.save_dir,
            batches_jsonl=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}.jsonl"),
            # summary_csv=os.path.join(save_dir, f"{self.dataset_name}_{self.index_name}_summary.csv"),
            summary_csv=os.path.join(self.save_dir, f"summary.csv"),
            config_yaml=os.path.join(self.save_dir, f"{self.dataset_name}_{self.index_name}_config.yaml"),
        )

        if overwrite:
            for p in [self.paths.batches_jsonl, self.paths.config_yaml]: 
                if os.path.exists(p):
                    os.remove(p)

        # ---- meters ----
        # item-weighted (avg over items in dataset)
        self.acc_item: Dict[str, AverageMeter] = {}
        self.perf_item: Dict[str, AverageMeter] = {}

        # batch-weighted (avg over batches)
        self.acc_batch: Dict[str, AverageMeter] = {}
        self.perf_batch: Dict[str, AverageMeter] = {}

        self.num_items = 0
        self.num_batches = 0

        self._fh_batches = open(self.paths.batches_jsonl, "a", encoding="utf-8") if self.log_batches else None
        self._since_flush = 0

    # -------- context manager --------
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------- internals --------
    def _get_meter(self, pool: Dict[str, AverageMeter], key: str) -> AverageMeter:
        if key not in pool:
            pool[key] = AverageMeter(key)
        return pool[key]

    def _write_jsonl(self, fh, obj: Dict[str, Any]) -> None:
        if fh is None:
            return
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._since_flush += 1
        if self.flush_every > 0 and self._since_flush >= self.flush_every:
            self.flush()

    # -------- filesystem --------
    def save_config(self, cfg) -> str:
        OmegaConf.save(config=cfg, f=self.paths.config_yaml, resolve=True)
        return self.paths.config_yaml

    def flush(self) -> None:
        if self._fh_batches is None:
            return
        self._fh_batches.flush()
        if self.fsync:
            os.fsync(self._fh_batches.fileno())
        self._since_flush = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._fh_batches is not None:
                self._fh_batches.close()
                self._fh_batches = None

    
    # -------- logging --------
    def log_batch(
        self,
        batch_id: int,
        items: List[Item],
        batch_perf_metrics: Optional[Dict[str, Any]] = None,
        batch_acc_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        batch_perf_metrics = batch_perf_metrics or {}
        batch_acc_metrics = batch_acc_metrics or {}
        bs = len(items)

        # ----- update batch-average meters (each batch counts once) -----
        for k, v in batch_acc_metrics.items():
            self._get_meter(self.acc_batch, k).update(v, n=1)
        for k, v in batch_perf_metrics.items():
            self._get_meter(self.perf_batch, k).update(v, n=1)

        # ----- update item-average meters (each item counts once) -----
         # Accuracy: your batch acc metrics are averages over items -> weight by batch size
        for k, v in batch_acc_metrics.items():
            self._get_meter(self.acc_item, k).update(v, n=bs)
        
        # Perf:for timings (totals per batch)
        for k, v in batch_perf_metrics.items():
            # store per-item derived metric too
            per_item_key = f"{k}_per_item"
            try:
                per_item_val = float(v) / max(1, bs)
                self._get_meter(self.perf_item, per_item_key).update(per_item_val, n=bs)
            except Exception:
                pass
        
        self.num_items += bs
        self.num_batches += 1

        if self.log_batches:
            # ----- write batch record -----
            record: Dict[str, Any] = {
                "timestamp": _now_iso(),
                "dataset_name": self.dataset_name,
                "pipeline_name": self.pipeline_name,
                "batch_id": int(batch_id),
                "batch_size": bs,
                "metrics": {
                    "perf_metrics": batch_perf_metrics,
                    "acc_metrics": batch_acc_metrics,
                },
            }

            if self.store_item_details_in_batch:
                record["items"] = [it.to_dict() for it in items]

            self._write_jsonl(self._fh_batches, record)


    # -------- progress reporting --------
    def maybe_report(self, pbar=None) -> None:
        if pbar is None:
            return
        if self.num_items == 0 or self.report_every_items <= 0:
            return
        if (self.num_items % self.report_every_items) != 0:
            return

        postfix: Dict[str, Any] = {"items": self.num_items, "batches": self.num_batches}

        # show item-avg acc metrics by default (usually what you care about)
        if "f1" in self.acc_item:
            postfix["f1"] = f"{self.acc_item['f1'].avg:.3f}"
        if "em" in self.acc_item:
            postfix["em"] = f"{self.acc_item['em'].avg:.3f}"

        key_mapping = {
            "encode_query_time(s)": "encode_q(s)",
            "search_time(s)": "search(s)",
            "generation_time(s)": "gen(s)",
            "cache_hit": "cache_hits",
        }

        for key in self.report_perf_keys:
            if key in self.perf_item:
                label = key_mapping.get(key, key)
                postfix[label] = f"{self.perf_item[key].avg:.2f}"

        pbar.set_postfix(postfix)

    # -------- finalize --------
    def finalize(self) -> Dict[str, Any]:
        self.close()

        summary: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "dataset": self.dataset_path,
            "dataset_name": self.dataset_name,
            "index_name": self.index_name,
            "index_path": self.index_path,
            "batches_jsonl": self.paths.batches_jsonl if self.log_batches else "",
            "pipeline": self.pipeline_name,
            "num_items": self.num_items,
            "num_batches": self.num_batches,
            "batch_size": self.cfg.batch_size,
            "retrieval_topk": (
                getattr(getattr(self.cfg, "retriever", None), "search", None).retrieval_topk
                if getattr(self.cfg, "retriever", None) is not None
                else ""
            ),
            "batches_jsonl": self.paths.batches_jsonl if self.log_batches else "",
            "config_yaml": self.paths.config_yaml,
        }

        

        # item-weighted outputs
        for k, m in self.acc_item.items():
            summary[f"avg_item.acc.{k}"] = m.avg
        for k, m in self.perf_item.items():
            summary[f"avg_item.perf.{k}"] = m.avg
            summary[f"sum_item.perf.{k}"] = m.sum

        # batch-weighted outputs
        for k, m in self.acc_batch.items():
            summary[f"avg_batch.acc.{k}"] = m.avg
        for k, m in self.perf_batch.items():
            summary[f"avg_batch.perf.{k}"] = m.avg
            summary[f"sum_batch.perf.{k}"] = m.sum

        file_exists = os.path.exists(self.paths.summary_csv) and os.path.getsize(self.paths.summary_csv) > 0
        with open(self.paths.summary_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

        return summary