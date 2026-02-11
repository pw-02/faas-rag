from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil
import torch

# -------------------------
# NVML (GPU stats) optional
# -------------------------
_NVML_AVAILABLE = False
_nvml_init_error: Optional[Exception] = None

try:
    from pynvml import (  # type: ignore
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetTemperature,
        NVML_TEMPERATURE_GPU,
    )

    nvmlInit()
    _NVML_AVAILABLE = True
except Exception as e:
    _NVML_AVAILABLE = False
    _nvml_init_error = e

# Reuse the same Process object across calls.
# This is important for cpu_percent(interval=None) to return meaningful values.
_PROC = psutil.Process()


def get_resource_snapshot(*, gpu_index: int = 0) -> dict[str, Any]:
    """
    Lightweight snapshot for observability (not billing-grade).

    Notes:
      - proc.cpu_percent(interval=None) is "since last call" *on this Process object*
      - psutil.cpu_percent(interval=None) is global "since last call"
      - Both require priming once before the first real sample
    """
    vm = psutil.virtual_memory()

    snap: dict[str, Any] = {
        # Process
        "proc_rss_gb": _PROC.memory_info().rss / (1024**3),
        "proc_cpu_percent": _PROC.cpu_percent(interval=None),

        # System
        "system_mem_total_gb": vm.total / (1024**3),
        "system_mem_available_gb": vm.available / (1024**3),
        "system_mem_used_gb": vm.used / (1024**3),
        "system_mem_percent": float(vm.percent),
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_cpu_cores": int(psutil.cpu_count(logical=True) or 0),
    }

    # GPU stats (best-effort)
    if torch.cuda.is_available() and _NVML_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(int(gpu_index))
            util = nvmlDeviceGetUtilizationRates(handle)
            mem = nvmlDeviceGetMemoryInfo(handle)
            temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)

            snap.update(
                {
                    "gpu_index": int(gpu_index),
                    "gpu_util_percent": float(util.gpu),
                    "gpu_mem_used_gb": float(mem.used) / (1024**3),
                    "gpu_mem_total_gb": float(mem.total) / (1024**3),
                    "gpu_temp_c": float(temp),
                }
            )
        except Exception:
            # Best-effort: ignore NVML failures (driver perms, transient errors)
            pass

    return snap


async def resource_monitor_loop(
    *,
    interval_s: float = 5.0,
    output_path: str = "resource_usage.jsonl",
    gpu_index: int = 0,
    print_banner: bool = True,
) -> None:
    """
    Periodically append absolute resource snapshots to a JSONL file.

    "Very common" production pattern:
      - prime cpu_percent counters once
      - reuse psutil.Process() across samples
      - open the output file once
      - support clean cancellation
    """
    interval_s = float(interval_s)
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")

    # Prime counters so the first sample isn't 0.0 due to "since last call" semantics.
    _PROC.cpu_percent(None)
    psutil.cpu_percent(None)

    if print_banner:
        if _NVML_AVAILABLE:
            print("NVML initialized; GPU stats will be included in resource snapshots.")
        else:
            msg = "NVML not available; GPU stats will be skipped in resource snapshots."
            if _nvml_init_error is not None:
                msg += f" (reason: {_nvml_init_error})"
            print(msg)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:  # <-- overwrite/truncate
            while True:
                snap = get_resource_snapshot(gpu_index=gpu_index)
                record = {
                    "ts": time.time(),
                    "ts_iso": datetime.now(timezone.utc).isoformat(),
                    **snap,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"Error in resource monitor loop: {e}")
        raise


    # try:
    #     with path.open("a", encoding="utf-8") as f:
    #         while True:
    #             snap = get_resource_snapshot(gpu_index=gpu_index)
    #             record = {
    #                 "ts": time.time(),
    #                 "ts_iso": datetime.now(timezone.utc).isoformat(),
    #                 **snap,
    #             }
    #             f.write(json.dumps(record) + "\n")
    #             f.flush()
    #             await asyncio.sleep(interval_s)
    # except asyncio.CancelledError:
    #     # Clean cancellation: just exit
    #     raise
    # except Exception as e:
    #     print(f"Error in resource monitor loop: {e}")
    #     raise