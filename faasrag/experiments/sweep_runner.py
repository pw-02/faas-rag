#!/usr/bin/env python3
"""
sweep.py

For each experiment:
  1) start service with Hydra overrides
  2) wait until ready (Ping RPC)
  3) run client to produce results.jsonl
  4) stop service
  5) repeat

Now supports sweeping CLIENT concurrency levels via --concurrency_levels.

Writes outputs under:
  runs/<exp_folder>/<idx>_<name>/cXXX/
    - service.log
    - client.log
    - results.jsonl
    - meta.json
    - (optional) resource_usage.jsonl + service_config.yaml (if telemetry enabled)
"""

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import grpc

import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc


SUPPORTED_INDEX_TYPES = {
    "100k": ["hnsw", "flat", "ivf"],
    "500k": ["flat", "hnsw", "ivf"],
    "1m": ["flat", "hnsw", "ivf"],
    "2_5m": ["hnsw", "ivf"],
    "5m": ["hnsw", "ivf"],
    "10m": ["hnsw"],
    "21m": ["hnsw"],
}

SUPPORTED_DOCSTORE_BACKENDS = [
    "local_sqlite",
    "local_jsonl_offsets",
    "s3_jsonl_offsets",
    "memory_jsonl",
]

ALL_INDEX_TYPES = sorted({t for types in SUPPORTED_INDEX_TYPES.values() for t in types})
ALL_SIZES = sorted(SUPPORTED_INDEX_TYPES.keys())
ALL_DOCSTORE_BACKENDS = list(SUPPORTED_DOCSTORE_BACKENDS)


def _make_experiment(index_type: str, dataset_size: str, docstore_backend: str, exp_folder_name: str) -> Dict[str, object]:
    return {
        "name": f"wiki_faiss_{index_type}_{dataset_size}_{docstore_backend}",
        "exp_folder_name": exp_folder_name,
        "overrides": [
            f"index=wiki_faiss_{index_type}_{dataset_size}",
            f"docstore=wiki_dpr_{dataset_size}",
            f"docstore_backend={docstore_backend}",
        ],
    }


def build_docstore_backend_experiments(
    *,
    index_type: str = "hnsw",
    dataset_size: str = "21m",
) -> List[Dict[str, object]]:
    if dataset_size not in SUPPORTED_INDEX_TYPES:
        raise ValueError(f"Unknown dataset_size {dataset_size!r}. Supported: {ALL_SIZES}")
    if index_type not in SUPPORTED_INDEX_TYPES[dataset_size]:
        raise ValueError(
            f"index_type {index_type!r} is not supported for dataset_size {dataset_size!r}. "
            f"Supported for {dataset_size}: {SUPPORTED_INDEX_TYPES[dataset_size]}"
        )
    return [_make_experiment(index_type, dataset_size, b, "docstore_backend_exps") for b in ALL_DOCSTORE_BACKENDS]


def build_index_type_experiments(
    *,
    dataset_size: str = "21m",
    docstore_backend: str = "local_jsonl_offsets",
) -> List[Dict[str, object]]:
    if dataset_size not in SUPPORTED_INDEX_TYPES:
        raise ValueError(f"Unknown dataset_size {dataset_size!r}. Supported: {ALL_SIZES}")
    if docstore_backend not in ALL_DOCSTORE_BACKENDS:
        raise ValueError(f"Unknown docstore_backend {docstore_backend!r}. Supported: {ALL_DOCSTORE_BACKENDS}")

    types = SUPPORTED_INDEX_TYPES[dataset_size]
    return [_make_experiment(t, dataset_size, docstore_backend, "index_type_exps") for t in types]


def build_index_size_experiments(
    *,
    index_type: str = "hnsw",
    docstore_backend: str = "local_jsonl_offsets",
) -> List[Dict[str, object]]:
    if index_type not in ALL_INDEX_TYPES:
        raise ValueError(f"Unknown index_type {index_type!r}. Supported: {ALL_INDEX_TYPES}")
    if docstore_backend not in ALL_DOCSTORE_BACKENDS:
        raise ValueError(f"Unknown docstore_backend {docstore_backend!r}. Supported: {ALL_DOCSTORE_BACKENDS}")

    exps: List[Dict[str, object]] = []
    for size in ALL_SIZES:
        if index_type in SUPPORTED_INDEX_TYPES[size]:
            exps.append(_make_experiment(index_type, size, docstore_backend, "index_size_exps"))
    return exps

def build_experiment_for_concurrency_sweep() -> List[Dict[str, object]]:
    return [_make_experiment("hnsw", "21m", "local_sqlite", "concurrency_exps")]

def build_experiments(
    index_type: Optional[str] = "hnsw",
    dataset_size: Optional[str] = "21m",
    docstore_backend: Optional[str] = None,
) -> List[Dict[str, object]]:
    sizes = [dataset_size] if dataset_size is not None else ALL_SIZES
    backends = [docstore_backend] if docstore_backend is not None else ALL_DOCSTORE_BACKENDS

    exps: List[Dict[str, object]] = []
    for size in sizes:
        if size not in SUPPORTED_INDEX_TYPES:
            raise ValueError(f"Unknown dataset_size {size!r}. Supported: {ALL_SIZES}")

        types = SUPPORTED_INDEX_TYPES[size]
        if index_type is not None:
            if index_type not in ALL_INDEX_TYPES:
                raise ValueError(f"Unknown index_type {index_type!r}. Supported: {ALL_INDEX_TYPES}")
            types = [t for t in types if t == index_type]

        for t in types:
            for b in backends:
                if b not in ALL_DOCSTORE_BACKENDS:
                    raise ValueError(f"Unknown docstore_backend {b!r}. Supported: {ALL_DOCSTORE_BACKENDS}")
                exps.append(_make_experiment(t, size, b, "general_experiments"))

    return exps


# -----------------------
# Subprocess helpers
# -----------------------

def _tee_stream(prefix, stream, logfile, *, to_console: bool):
    for line in iter(stream.readline, ""):
        msg = f"{prefix}{line}"
        if to_console:
            sys.stdout.write(msg)
            sys.stdout.flush()
        logfile.write(msg)
        logfile.flush()
    stream.close()


def popen_tee(
    cmd,
    *,
    log_path: Path,
    cwd=None,
    env=None,
    prefix: str = "",
    to_console: bool = True,
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")

    preexec_fn = None
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = os.setsid

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=preexec_fn,
        creationflags=creationflags,
    )

    threading.Thread(
        target=_tee_stream,
        args=(prefix, proc.stdout, log_f),
        kwargs={"to_console": to_console},
        daemon=True,
    ).start()

    return proc, log_f


def run_tee(cmd, *, log_path: Path, prefix: str = "", to_console: bool = True) -> int:
    proc, log_f = popen_tee(cmd, log_path=log_path, prefix=prefix, to_console=to_console)
    try:
        return proc.wait()
    finally:
        try:
            log_f.close()
        except Exception:
            pass


def stop_process_tree(proc: subprocess.Popen, *, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < grace_s:
            if proc.poll() is not None:
                return
            time.sleep(0.2)

        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    else:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

        t0 = time.time()
        while time.time() - t0 < grace_s:
            if proc.poll() is not None:
                return
            time.sleep(0.2)

        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# -----------------------
# Readiness check (Ping)
# -----------------------

async def wait_for_service_ready(target: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None

    while time.time() < deadline:
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = rag_pb2_grpc.RAGServiceStub(channel)
                resp = await stub.Ping(rag_pb2.PingRequest(), timeout=1.0)
                if getattr(resp, "ok", False):
                    return
                last_err = "Ping returned ok=false"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        await asyncio.sleep(2)

    raise TimeoutError(f"Service not ready at {target} after {timeout_s}s. Last error: {last_err}")


def wait_for_service_ready_sync(target: str, timeout_s: float) -> None:
    asyncio.run(wait_for_service_ready(target, timeout_s=timeout_s))


# -----------------------
# Run one experiment (one concurrency point)
# -----------------------

def run_experiment(
    *,
    idx: int,
    name: str,
    overrides: List[str],
    args,
    service_base: List[str],
    client_base: List[str],
    run_dir: Path,
    client_concurrency: int,
) -> None:
    port = args.base_port + idx
    target = f"{args.host}:{port}"

    service_log = run_dir / "service.log"
    client_log = run_dir / "client.log"
    results_path = run_dir / "results.jsonl"
    meta_path = run_dir / "meta.json"

    if args.skip_if_exists and results_path.exists():
        print(f"\n=== SKIP {idx:02d}: {name} c={client_concurrency} (results.jsonl exists) ===")
        return

    # Clean dir if rerunning a failed attempt
    if run_dir.exists() and not results_path.exists():
        for f in run_dir.iterdir():
            if f.is_file():
                f.unlink()

    telemetry_overrides: List[str] = []
    if args.enable_telemetry:
        telemetry_overrides = [
            "telemetry.enabled=true",
            "telemetry.interval_s=2",
            f"telemetry.dir={run_dir}",
        ]

    service_cmd = service_base + [f"host={args.host}", f"port={port}"] + telemetry_overrides + overrides

    client_cmd = (
        client_base
        + ["--target", target]
        + ["--dataset_path", args.dataset_path]
        + ["--limit", str(args.limit)]
        + ["--deadline_s", str(args.deadline_s)]
        + ["--concurrency", str(client_concurrency)]
        + ["--retries", str(args.retries)]
        + ["--retry_backoff_s", str(args.retry_backoff_s)]
        + ["--seed", str(args.seed)]
        + ["--out", str(results_path)]
    )
    if args.shuffle:
        client_cmd.append("--shuffle")

    print(f"\n=== RUN {idx:02d}: {name} c={client_concurrency} @ {target} ===")
    print("SERVICE:", " ".join(service_cmd))
    print("CLIENT: ", " ".join(client_cmd))

    t_start = time.time()
    service_proc, service_log_f = popen_tee(
        service_cmd, log_path=service_log, prefix="[SERVICE] ", to_console=not args.quiet
    )

    try:
        wait_for_service_ready_sync(target, timeout_s=float(args.ready_timeout_s))

        ret = run_tee(client_cmd, log_path=client_log, prefix="[CLIENT] ", to_console=not args.quiet)
        if ret != 0:
            raise RuntimeError(f"Client failed with exit code {ret}")

        wall_s = time.time() - t_start
        meta = {
            "run_name": name,
            "run_index": idx,
            "client_concurrency": client_concurrency,
            "target": target,
            "service_cmd": service_cmd,
            "client_cmd": client_cmd,
            "service_overrides": overrides,
            "dataset_path": args.dataset_path,
            "limit": args.limit,
            "client_params": {
                "deadline_s": args.deadline_s,
                "concurrency": client_concurrency,
                "retries": args.retries,
                "retry_backoff_s": args.retry_backoff_s,
                "shuffle": args.shuffle,
                "seed": args.seed,
            },
            "telemetry": {"enabled": bool(args.enable_telemetry), "dir": str(run_dir) if args.enable_telemetry else None},
            "wall_time_s": wall_s,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "outputs": {
                "service_log": str(service_log),
                "client_log": str(client_log),
                "results_jsonl": str(results_path),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    finally:
        stop_process_tree(service_proc)
        try:
            service_log_f.close()
        except Exception:
            pass


# -----------------------
# Main
# -----------------------

def _parse_concurrency_levels(s: str) -> List[int]:
    levels: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError as e:
            raise ValueError(f"Bad --concurrency_levels entry {part!r} (must be int)") from e
        if v < 1:
            raise ValueError(f"Concurrency must be >= 1, got {v}")
        levels.append(v)
    # de-dupe while preserving order
    seen = set()
    out = []
    for v in levels:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--service_cmd", default="python -u -m faasrag.server.server")
    ap.add_argument("--client_cmd", default="python -u -m faasrag.client.nq_rag_client")

    ap.add_argument("--runs_dir", default="runs", help="Directory to store run outputs")
    ap.add_argument("--dataset_path", default="data/datasets/qa/nq/nq_dev.jsonl")
    ap.add_argument("--limit", type=int, default=500)

    ap.add_argument("--base_port", type=int, default=50051)
    ap.add_argument("--host", default="127.0.0.1")

    ap.add_argument("--deadline_s", type=float, default=3000.0)
    ap.add_argument("--concurrency_levels", type=str, default="1,2,4,8,16",
                    help="Comma-separated client concurrency levels to sweep (e.g. '1,2,4,8,16')")

    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry_backoff_s", type=float, default=0.5)
    ap.add_argument("--shuffle", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--ready_timeout_s", type=float, default=12000.0)
    ap.add_argument("--enable_telemetry", action="store_true", default=True)
    ap.add_argument("--skip_if_exists", action="store_true", default=False)

    ap.add_argument("--quiet", action="store_true", default=True,
                    help="Do not print service/client logs to console (file only)")

    args = ap.parse_args()
    concurrency_levels = _parse_concurrency_levels(args.concurrency_levels)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    service_base = shlex.split(args.service_cmd)
    client_base = shlex.split(args.client_cmd)

    # Choose which experiments to run:
    # experiments = build_experiments()
    # experiments = build_docstore_backend_experiments()
    # experiments = build_index_size_experiments()
    # experiments += build_index_type_experiments()
    experiments = build_experiment_for_concurrency_sweep()

    print(f"Total experiments to run: {len(experiments) * len(concurrency_levels)} ({len(experiments)} experiment configs x {len(concurrency_levels)} concurrency levels)")
    print(f"Concurrency levels: {concurrency_levels}")
    print("Experiments:")
    for idx, exp in enumerate(experiments):
        print(f"  {idx:02d}: {exp['name']}")

    for idx, exp in enumerate(experiments):
        name = str(exp["name"])
        overrides = list(exp.get("overrides", []))
        exp_folder_name = str(exp.get("exp_folder_name", "other_experiments"))

        for c in concurrency_levels:
            run_dir = runs_dir / exp_folder_name / f"{idx:02d}_{name}" / f"c{c:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_experiment(
                idx=idx,
                name=name,
                overrides=overrides,
                args=args,
                service_base=service_base,
                client_base=client_base,
                run_dir=run_dir,
                client_concurrency=c,
            )

    print("\n✅ Sweep complete.")


if __name__ == "__main__":
    main()
