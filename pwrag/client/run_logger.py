import os
import json
import csv
from typing import Any, Dict, Optional
from datetime import datetime
from pwrag.args.args import AppConfig
from omegaconf import OmegaConf


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        try:
            v = float(val)
        except Exception:
            return
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class RunLogger:
    def __init__(
        self,
        config: AppConfig,
        pipeline_name: Optional[str] = "",
        overwrite: bool = True,
        report_every: int = 10,
        log_items: bool = True,
        run_name: Optional[str] = None,
        flush_every: int = 50,
        fsync: bool = False,
    ):
       # run name
        self.run_config:AppConfig = config
        self.pipeline_name = pipeline_name
        self.log_items = bool(log_items)
        self.flush_every = int(flush_every)
        self.fsync = bool(fsync)

        # self.save_dir = os.path.join(
        #     config.save_dir,
        #     config.dataset.dataset_name,
        #     config.generator.name,
        #     self.pipeline_name,
        #     # datetime.now().strftime("%Y%m%d_%H%M%S")
        # )
        self.save_dir =  config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # self.run_name = run_name or f"run_{config.dataset.dataset_name}_{config.generator.name}"
        self.run_name = run_name or f"items"
        self.jsonl_path = os.path.join(self.save_dir, f"{self.run_name}.jsonl")
        self.summary_csv = os.path.join(self.save_dir, f"{self.run_name}_summary.csv")

        if overwrite:
            if os.path.exists(self.summary_csv):
                os.remove(self.summary_csv)
            if self.log_items and os.path.exists(self.jsonl_path):
                os.remove(self.jsonl_path)

        self.report_every = int(report_every)
        self.n = 0
        self._since_flush = 0

        self.acc_meters: Dict[str, AverageMeter] = {}
        self.cost_meters: Dict[str, AverageMeter] = {}
        self.em_correct = 0

        # keep file handle open
        self._fh = None
        if self.log_items:
            self._fh = open(self.jsonl_path, "a", encoding="utf-8")
    
    def save_config(self, cfg) -> str:
        """Save resolved Hydra config next to the logs. Returns path."""
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"config.yaml")
        OmegaConf.save(config=cfg, f=path, resolve=True)
        return path

    def flush(self) -> None:
        """Flush buffered JSONL writes to disk (and optionally fsync)."""
        if not self.log_items or self._fh is None:
            return
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())
        self._since_flush = 0

    def close(self) -> None:
        if self._fh is not None:
            try:
                self.flush()
            finally:
                self._fh.close()
                self._fh = None

    def _get_meter(self, pool: Dict[str, "AverageMeter"], key: str):
        if key not in pool:
            pool[key] = AverageMeter(key)
        return pool[key]

    def log_item(self, record: Dict[str, Any]):
        # write JSONL (optional)
        if self.log_items and self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._since_flush += 1
            if self.flush_every > 0 and self._since_flush >= self.flush_every:
                self.flush()

        self.n += 1

        metrics = record.get("metrics", {}) or {}
        acc = metrics.get("acc_metrics", {}) or {}
        cost = metrics.get("metrics", {}) or {}

        if isinstance(acc, dict):
            for k, v in acc.items():
                self._get_meter(self.acc_meters, k).update(v)

        if isinstance(cost, dict):
            for k, v in cost.items():
                self._get_meter(self.cost_meters, k).update(v)

        try:
            if float(acc.get("em", 0)) >= 1.0:
                self.em_correct += 1
        except Exception:
            pass

    def maybe_report(self, pbar=None):
        if self.n == 0 or self.report_every <= 0 or (self.n % self.report_every != 0):
            return

        postfix = {}
        # if "em" in self.acc_meters:
        #     postfix["em"] = f"{self.acc_meters['em'].avg:.3f}"
        if "f1" in self.acc_meters:
            postfix["f1"] = f"{self.acc_meters['f1'].avg:.3f}"

        key_mapping = {
            "encode_query_time(s)": "encode_q(s)",
            "search_time(s)": "search(s)",
            "generation_time(s)": "gen(s)",
            "cache_hit": "cache_hits",
        }
        
        for key in ["encode_query_time(s)", "search_time(s)", "generation_time(s)"]:
            if key in self.cost_meters:
                postfix[key_mapping[key]] = f"{self.cost_meters[key].avg:.2f}"
        # for k, m in self.cost_meters.items():
        #     postfix[k] = f"{m.avg:.2f}"
        postfix["n"] = self.n

        if pbar is not None:
            pbar.set_postfix(postfix)

    def finalize(self):
        # ensure file is flushed/closed
        self.close()

        summary = {
            "run_name": self.run_name,
            "dataset": self.run_config.dataset.dataset_path,
            "num_samples": self.n,
            "pipeline": self.pipeline_name,
            "retrieval_topk": self.run_config.retriever.search.retrieval_topk,
            "jsonl_path": self.jsonl_path if self.log_items else "",
        }

        for k, m in self.acc_meters.items():
            summary[f"acc_avg.{k}"] = m.avg
            # summary[f"acc_sum.{k}"] = m.sum

        for k, m in self.cost_meters.items():
            summary[f"cost_avg.{k}"] = m.avg
            summary[f"cost_sum.{k}"] = m.sum

        with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        return summary