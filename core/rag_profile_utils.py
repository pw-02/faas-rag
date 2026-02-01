from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch

try:
    import psutil
except ImportError:
    psutil = None

# NVML (GPU utilization / total VRAM)
try:
    import pynvml
    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False

# Unix-only true peak RSS (since process start)
try:
    import resource
    _RESOURCE_OK = True
except Exception:
    resource = None
    _RESOURCE_OK = False


# -----------------------------
# NVML lifecycle (refcounted)
# -----------------------------
_NVML_REFCOUNT = 0


def _nvml_init() -> bool:
    """Initialize NVML once per process (refcounted)."""
    global _NVML_REFCOUNT
    if not _NVML_OK:
        return False
    try:
        if _NVML_REFCOUNT == 0:
            pynvml.nvmlInit()
        _NVML_REFCOUNT += 1
        return True
    except Exception:
        return False


def _nvml_shutdown() -> None:
    """Shutdown NVML when last user releases it (refcounted)."""
    global _NVML_REFCOUNT
    if not _NVML_OK:
        return
    try:
        if _NVML_REFCOUNT > 0:
            _NVML_REFCOUNT -= 1
            if _NVML_REFCOUNT == 0:
                pynvml.nvmlShutdown()
    except Exception:
        # If shutdown fails, don't crash; leave refcount as-is.
        pass


# -----------------------------
# Resource stats
# -----------------------------
@dataclass
class ResourceSnapshot:
    # process CPU / RAM
    rss_gb: Optional[float] = None
    peak_rss_gb: Optional[float] = None
    cpu_percent: Optional[float] = None

    # system CPU
    system_cpu_percent: Optional[float] = None
    system_num_cpus: Optional[int] = None

    # system RAM
    system_mem_total_gb: Optional[float] = None
    system_mem_available_gb: Optional[float] = None
    system_mem_used_gb: Optional[float] = None
    system_mem_percent: Optional[float] = None

    # GPU (torch memory)
    gpu_allocated_gb: Optional[float] = None
    gpu_reserved_gb: Optional[float] = None
    gpu_peak_allocated_gb: Optional[float] = None

    # GPU (NVML)
    gpu_util_percent: Optional[float] = None
    gpu_mem_used_gb: Optional[float] = None
    gpu_mem_total_gb: Optional[float] = None


class ResourceMonitor:
    """
    Per-run resource monitor.
    - Tracks a best-effort peak RSS by taking the max of observed RSS samples.
    - Uses Unix `resource` (ru_maxrss) when available for a truer peak.
    - Uses NVML for GPU utilization and total VRAM if available.
    """

    def __init__(self, device: str | torch.device, gpu_index: int = 0):
        self.device = str(device).strip().lower()
        self.gpu_index = int(gpu_index)
        self.peak_rss_gb_seen: float = 0.0

        self._nvml_inited = False
        self._nvml_handle = None

        # Prime psutil cpu_percent so first snapshot isn't always 0.0
        if psutil is not None:
            try:
                p = psutil.Process(os.getpid())
                p.cpu_percent(interval=None)
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        # Initialize NVML once (refcounted) if CUDA requested
        if self.device.startswith("cuda") and _NVML_OK:
            if _nvml_init():
                try:
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    self._nvml_inited = True
                except Exception:
                    self._nvml_handle = None
                    self._nvml_inited = False
                    _nvml_shutdown()

    def close(self) -> None:
        if self._nvml_inited:
            self._nvml_inited = False
            self._nvml_handle = None
            _nvml_shutdown()

    def reset_torch_gpu_peak(self) -> None:
        """Reset torch CUDA peak counters so gpu_peak_allocated_gb is per-interval peak."""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    @staticmethod
    def _ru_maxrss_to_gb(ru_maxrss: float) -> float:
        """
        Convert ru_maxrss to GB.
        Common behavior:
        - Linux: kilobytes
        - macOS: bytes
        """
        if sys.platform == "darwin":
            # bytes -> GB
            return float(ru_maxrss) / (1024**3)
        # Assume kilobytes -> GB
        return (float(ru_maxrss) * 1024.0) / (1024**3)

    def snapshot(self) -> dict[str, Any]:
        snap = ResourceSnapshot()

        # ----- CPU / RAM (process + system) -----
        if psutil is not None:
            try:
                p = psutil.Process(os.getpid())

                rss_gb = p.memory_info().rss / (1024**3)
                snap.rss_gb = rss_gb

                # NOTE: cpu_percent is since last call; primed in __init__
                snap.cpu_percent = p.cpu_percent(interval=None)

                # portable peak by tracking max observed
                self.peak_rss_gb_seen = max(self.peak_rss_gb_seen, rss_gb)
                snap.peak_rss_gb = self.peak_rss_gb_seen

                # system-wide memory
                vm = psutil.virtual_memory()
                snap.system_mem_total_gb = vm.total / (1024**3)
                snap.system_mem_available_gb = vm.available / (1024**3)
                snap.system_mem_used_gb = vm.used / (1024**3)
                snap.system_mem_percent = float(vm.percent)

                snap.system_cpu_percent = psutil.cpu_percent(interval=None)
                snap.system_num_cpus = psutil.cpu_count(logical=True)

            except Exception:
                pass

        # More "true" peak RSS on Unix: ru_maxrss
        if _RESOURCE_OK:
            try:
                ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                peak_gb = self._ru_maxrss_to_gb(ru)
                snap.peak_rss_gb = max(snap.peak_rss_gb or 0.0, peak_gb)
            except Exception:
                pass

        # ----- GPU (torch) -----
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                snap.gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                snap.gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                snap.gpu_peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                pass

        # ----- GPU (NVML utilization + total VRAM used) -----
        if self._nvml_inited and self._nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                snap.gpu_util_percent = float(util.gpu)
                snap.gpu_mem_used_gb = mem.used / (1024**3)
                snap.gpu_mem_total_gb = mem.total / (1024**3)
            except Exception:
                pass

        return asdict(snap)


# -----------------------------
# CSV outputs
# -----------------------------
def save_batch_results_csv(results: list[dict[str, Any]], path: str) -> None:
    """Save batch-level results (one row per batch)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in results:
        timings = r.get("timings", {}) or {}
        resources = (r.get("resources_after") or r.get("resources") or {}) or {}

        batch_size = int(r.get("batch_size", 0) or 0)
        total_s = float(timings.get("total_s", 0.0) or 0.0)

        rows.append({
            # metadata
            "batch_start_idx": int(r.get("batch_start", 0) or 0),
            "batch_size": batch_size,
            "top_k": int(r.get("top_k", 0) or 0),
            "max_context_docs": int(r.get("max_context_docs", 0) or 0),

            # timings
            "embed_time_s": float(timings.get("embed_s", 0.0) or 0.0),
            "search_time_s": float(timings.get("search_s", 0.0) or 0.0),
            "docstore_fetch_time_s": float(timings.get("docstore_s", 0.0) or 0.0),
            "prompt_build_time_s": float(timings.get("prompt_s", 0.0) or 0.0),
            "generation_time_s": float(timings.get("generate_s", 0.0) or 0.0),
            "total_time_s": total_s,

            # throughput
            "throughput_qps": (batch_size / total_s) if total_s > 0 else 0.0,
            "avg_latency_per_query_s": (total_s / batch_size) if batch_size > 0 else 0.0,

            # resources (0 if missing)
            "rss_gb": float(resources.get("rss_gb") or 0.0),
            "peak_rss_gb": float(resources.get("peak_rss_gb") or 0.0),
            "cpu_percent": float(resources.get("cpu_percent") or 0.0),

            # system RAM
            "system_mem_total_gb": float(resources.get("system_mem_total_gb") or 0.0),
            "system_mem_available_gb": float(resources.get("system_mem_available_gb") or 0.0),
            "system_mem_used_gb": float(resources.get("system_mem_used_gb") or 0.0),
            "system_mem_percent": float(resources.get("system_mem_percent") or 0.0),

            # system CPU
            "system_cpu_percent": float(resources.get("system_cpu_percent") or 0.0),
            "system_num_cpus": int(resources.get("system_num_cpus") or 0),

            # GPU
            "gpu_allocated_gb": float(resources.get("gpu_allocated_gb") or 0.0),
            "gpu_reserved_gb": float(resources.get("gpu_reserved_gb") or 0.0),
            "gpu_peak_allocated_gb": float(resources.get("gpu_peak_allocated_gb") or 0.0),

            # NVML
            "gpu_util_percent": float(resources.get("gpu_util_percent") or 0.0),
            "gpu_mem_used_gb": float(resources.get("gpu_mem_used_gb") or 0.0),
            "gpu_mem_total_gb": float(resources.get("gpu_mem_total_gb") or 0.0),
        })

    if not rows:
        return

    # Future-proof: union of keys across rows so we don't silently drop later keys
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})

    with path_obj.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def create_summary_from_csvs(csv_paths: list[str], summary_csv_output_path: str) -> pd.DataFrame:
    """Summarize batch-level CSVs and write a summary CSV."""
    summary_rows: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        p = Path(csv_path)
        df = pd.read_csv(p)
        if df.empty:
            continue

        # Batch size stats (don't assume constant)
        if "batch_size" in df.columns and not df["batch_size"].empty:
            bs = df["batch_size"].astype(int)
            batch_size_mode = int(bs.mode().iloc[0]) if not bs.mode().empty else int(bs.iloc[0])
            min_batch_size = int(bs.min())
            max_batch_size = int(bs.max())
        else:
            batch_size_mode = 0
            min_batch_size = 0
            max_batch_size = 0

        total_queries = int(df["batch_size"].sum()) if "batch_size" in df.columns else 0
        total_time = float(df["total_time_s"].sum()) if "total_time_s" in df.columns else 0.0
        overall_qps = (total_queries / total_time) if total_time > 0 else 0.0

        summary_rows.append({
            "index": p.stem,
            "batch_size_mode": batch_size_mode,
            "min_batch_size": min_batch_size,
            "max_batch_size": max_batch_size,
            "num_batches": int(len(df)),
            "total_queries": total_queries,

            "avg_total_time_s": float(df["total_time_s"].mean()) if "total_time_s" in df.columns else 0.0,
            "p95_total_time_s": float(df["total_time_s"].quantile(0.95)) if "total_time_s" in df.columns else 0.0,
            "overall_qps": overall_qps,

            # resource aggregates (if present)
            "avg_rss_gb": float(df["rss_gb"].mean()) if "rss_gb" in df.columns else 0.0,
            "avg_peak_rss_gb": float(df["peak_rss_gb"].mean()) if "peak_rss_gb" in df.columns else 0.0,
            "avg_system_mem_available_gb": float(df["system_mem_available_gb"].mean()) if "system_mem_available_gb" in df.columns else 0.0,
            "min_system_mem_available_gb": float(df["system_mem_available_gb"].min()) if "system_mem_available_gb" in df.columns else 0.0,
            "avg_gpu_util_percent": float(df["gpu_util_percent"].mean()) if "gpu_util_percent" in df.columns else 0.0,
            "avg_gpu_peak_allocated_gb": float(df["gpu_peak_allocated_gb"].mean()) if "gpu_peak_allocated_gb" in df.columns else 0.0,
        })

    summary_df = pd.DataFrame(summary_rows)
    out = Path(summary_csv_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)
    return summary_df


# -----------------------------
# Query loader (JSONL)
# -----------------------------
def load_questions_from_jsonl(
    path: str,
    *,
    column: str = "question",
    batch_size: int = 1,
    max_batches: int | None = None,
) -> list[str]:
    df = pd.read_json(path, lines=True)

    if column not in df.columns:
        raise KeyError(f"Column {column!r} not found in {path}")

    questions = df[column].dropna().astype(str).str.strip()
    questions = questions[questions != ""]  # drop empties after strip

    if max_batches is not None:
        max_queries = max_batches * batch_size
        questions = questions.head(max_queries)

    return questions.tolist()
