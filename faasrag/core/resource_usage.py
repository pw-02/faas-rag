from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timezone

import psutil
import torch

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
    )

    nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def get_resource_snapshot() -> dict[str, float]:
    """
    Return absolute process + system + GPU resource usage.

    Intended for observability / debugging (not billing-grade attribution).
    """

    proc = psutil.Process()
    vm = psutil.virtual_memory()

    snap: dict[str, float] = {
        # Process
        "proc_rss_mb": proc.memory_info().rss / 1024 / 1024,
        "proc_cpu_percent": proc.cpu_percent(interval=None),

        # System
        "system_mem_total_mb": vm.total / 1024 / 1024,
        "system_mem_available_mb": vm.available / 1024 / 1024,
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_cpu_cores": psutil.cpu_count(logical=True),
    }

    # GPU stats (best-effort)
    if torch.cuda.is_available() and _NVML_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(0)

            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

            snap.update(
                {
                    "gpu_util_percent": float(util.gpu),
                    "gpu_mem_used_mb": mem.used / 1024 / 1024,
                    "gpu_mem_total_mb": mem.total / 1024 / 1024,
                    "gpu_temp_c": float(temp),
                }
            )
        except Exception:
            pass

    return snap


async def resource_monitor_loop(
    interval_s: float = 5.0,
    output_path: str = "resource_usage.jsonl",
):
    """
    Periodically write absolute resource snapshots to jsonl.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        snap = get_resource_snapshot()

        record = {
            "ts": time.time(),
            "ts_iso": datetime.now(timezone.utc).isoformat(),
            **snap,
        }

        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        await asyncio.sleep(interval_s)
