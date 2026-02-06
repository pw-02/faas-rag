#!/usr/bin/env python3
"""
sweep.py

Run a sweep of RAG service configurations (e.g., different indexes), and for each:
  1) start the service with Hydra overrides
  2) wait until it is ready (Ping RPC)
  3) run your NQ JSONL client to produce results.jsonl
  4) stop the service
  5) repeat

Notes:
- DOES NOT override service artifact_dir (per your request).
- Writes per-run logs + client outputs under runs/<run_name>/.
- Uses a unique port per run to avoid port-reuse issues.
- Cross-platform process handling (Windows + POSIX).

Example:
  python sweep.py \
    --service_cmd "python -m faasrag.server.server" \
    --client_cmd "python -m faasrag.client.nq_rag_client" \
    --dataset_path data/datasets/qa/nq/nq_dev.jsonl \
    --limit 500 \
    --runs_dir runs \
    --base_port 50051
"""

import argparse
import asyncio
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import grpc

import faasrag.protos.rag_pb2 as rag_pb2
import faasrag.protos.rag_pb2_grpc as rag_pb2_grpc


# -----------------------
# Process management
# -----------------------

def _popen_with_logs(
    cmd: List[str],
    *,
    stdout_path: Path,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[subprocess.Popen, "object"]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(stdout_path, "w", encoding="utf-8")

    preexec_fn = None
    creationflags = 0

    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        preexec_fn = os.setsid  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=preexec_fn,
        creationflags=creationflags,
    )
    return proc, log_f


def _stop_process_tree(proc: subprocess.Popen, *, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        # Try graceful first (may be ignored depending on how it's launched)
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < grace_s:
            if proc.poll() is not None:
                return
            time.sleep(0.2)

        # Hard kill entire tree
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
        # POSIX
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

async def _wait_for_service_ready(target: str, timeout_s: float = 120.0) -> None:
    """
    Poll the service by calling Ping() with a short timeout until it responds with ok=True.
    """
    #set no timeout for now
    
    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    
    #set no timeout for now
    while True: #time.time() < deadline:
        try:
            async with grpc.aio.insecure_channel(target) as channel:
                stub = rag_pb2_grpc.RAGServiceStub(channel)
                resp = await stub.Ping(rag_pb2.PingRequest(), timeout=1.0)

                if getattr(resp, "ok", False):
                    return

                last_err = "Ping returned ok=false"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(5)

    raise TimeoutError(
        f"Service not ready at {target} after {timeout_s}s. Last error: {last_err}"
    )


def wait_for_service_ready_sync(target: str, timeout_s: float) -> None:
    """
    Synchronous wrapper (useful in environments where you don't want to manage an event loop).
    """
    asyncio.run(_wait_for_service_ready(target, timeout_s=timeout_s))


# -----------------------
# Experiment definition
# -----------------------

def build_experiments() -> List[Dict[str, object]]:
    return [
        {"name": "wiki_faiss_flat_100k",  "overrides": ["index=wiki_faiss_flat_100k", "docstore=wiki_dpr_100k"]},
        {"name": "wiki_faiss_ivf_100k",   "overrides": ["index=wiki_faiss_ivf_100k", "docstore=wiki_dpr_100k"]},
        {"name": "wiki_faiss_hnsw_100k",  "overrides": ["index=wiki_faiss_hnsw_100k", "docstore=wiki_dpr_100k"]},
        
        {"name": "wiki_faiss_flat_500k",  "overrides": ["index=wiki_faiss_flat_500k", "docstore=wiki_dpr_500k"]},
        {"name": "wiki_faiss_hnsw_500k",  "overrides": ["index=wiki_faiss_hnsw_500k", "docstore=wiki_dpr_500k"]},
        {"name": "wiki_faiss_ivf_500k",  "overrides": ["index=wiki_faiss_ivf_500k", "docstore=wiki_dpr_500k"]},

        {"name": "wiki_faiss_flat_1m",  "overrides": ["index=wiki_faiss_flat_1m", "docstore=wiki_dpr_1m"]},
        {"name": "wiki_faiss_hnsw_1m",  "overrides": ["index=wiki_faiss_hnsw_1m", "docstore=wiki_dpr_1m"]},
        {"name": "wiki_faiss_ivf_1m",  "overrides": ["index=wiki_faiss_ivf_1m", "docstore=wiki_dpr_1m"]},

        {"name": "wiki_faiss_hnsw_2_5m",  "overrides": ["index=wiki_faiss_hnsw_2_5m", "docstore=wiki_dpr_2_5m"]},
        {"name": "wiki_faiss_ivf_2_5m",  "overrides": ["index=wiki_faiss_ivf_2_5m", "docstore=wiki_dpr_2_5m"]},

        
        {"name": "wiki_faiss_hnsw_5m",  "overrides": ["index=wiki_faiss_hnsw_5m", "docstore=wiki_dpr_5m"]},
        {"name": "wiki_faiss_ivf_5m",  "overrides": ["index=wiki_faiss_ivf_5m", "docstore=wiki_dpr_5m"]},
       
        # {"name": "wiki_faiss_hnsw_21m",  "overrides": ["index=wiki_faiss_hnsw_21m", "docstore=wiki_dpr_21m"]},
        # {"name": "wiki_faiss_hnsw_10m",  "overrides": ["index=wiki_faiss_hnsw_10m", "docstore=wiki_dpr_10m"]},

        #not yet created!

        # {"name": "wiki_faiss_flat_5m",  "overrides": ["index=wiki_faiss_flat_5m", "docstore=wiki_dpr_5m"]},

        # {"name": "wiki_faiss_flat_2_5m",  "overrides": ["index=wiki_faiss_flat_1m", "docstore=wiki_dpr_2_5m"]},

        # {"name": "wiki_faiss_flat_10m",  "overrides": ["index=wiki_faiss_flat_10m", "docstore=wiki_dpr_10m"]},
        # {"name": "wiki_faiss_ivf_10m",  "overrides": ["index=wiki_faiss_ivf_10m", "docstore=wiki_dpr_10m"]},

        # {"name": "wiki_faiss_flat_21m",  "overrides": ["index=wiki_faiss_flat_21m", "docstore=wiki_dpr_21m"]},
        # {"name": "wiki_faiss_ivf_21m",  "overrides": ["index=wiki_faiss_ivf_21m", "docstore=wiki_dpr_21m"]},

    ]


# -----------------------
# Runner
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--service_cmd",
        default="python -m faasrag.server.server",
        help='Command to start service (Hydra app), e.g. "python -m faasrag.server.server"',
    )
    ap.add_argument(
        "--client_cmd",
        default="python -m faasrag.client.nq_rag_client",
        help='Command to start client, e.g. "python -m faasrag.client.nq_rag_client"',
    )

    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--dataset_path", default="data/datasets/qa/nq/nq_dev.jsonl")
    ap.add_argument("--limit", type=int, default=200)

    ap.add_argument("--base_port", type=int, default=50051)
    ap.add_argument("--host", default="127.0.0.1")

    # client knobs (forwarded)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--deadline_s", type=float, default=3000.0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--retry_backoff_s", type=float, default=0.5)
    ap.add_argument("--shuffle", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)

    # readiness + telemetry wiring
    ap.add_argument("--ready_timeout_s", type=float, default=120.0)
    ap.add_argument("--enable_telemetry", action="store_true", default=False)

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    service_base = shlex.split(args.service_cmd)
    client_base = shlex.split(args.client_cmd)

    experiments = build_experiments()

    for idx, exp in enumerate(experiments):
        name = str(exp["name"])
        overrides = list(exp.get("overrides", []))

        run_dir = runs_dir / f"{idx:02d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        port = args.base_port + idx
        target = f"{args.host}:{port}"

        service_log = run_dir / "service.log"
        client_log = run_dir / "client.log"
        results_path = run_dir / "results.jsonl"
        meta_path = run_dir / "meta.json"

        telemetry_overrides: List[str] = []
        resource_usage_path = run_dir / "resource_usage.jsonl"
        telemetry_overrides = [
            "telemetry.enabled=true",
            "telemetry.interval_s=2",
            f"telemetry.path={resource_usage_path}",
        ]
        service_cmd = (
            service_base
            + [f"host={args.host}", f"port={port}"]
            + telemetry_overrides
            + overrides
        )

        print(f"\n=== RUN {idx:02d}: {name} @ {target} ===")
        print("SERVICE:", " ".join(service_cmd))

        service_proc, service_log_f = _popen_with_logs(service_cmd, stdout_path=service_log)

        t_start = time.time()
        try:
            # Use the sync wrapper (avoids creating/tearing down event loops in some debuggers)
            wait_for_service_ready_sync(target, timeout_s=float(args.ready_timeout_s))

            client_cmd = (
                client_base
                + ["--target", target]
                + ["--dataset_path", args.dataset_path]
                + ["--limit", str(args.limit)]
                + ["--k", str(args.k)]
                + ["--max_tokens", str(args.max_tokens)]
                + ["--deadline_s", str(args.deadline_s)]
                + ["--concurrency", str(args.concurrency)]
                + ["--retries", str(args.retries)]
                + ["--retry_backoff_s", str(args.retry_backoff_s)]
                + ["--seed", str(args.seed)]
                + ["--out", str(results_path)]
            )
            if args.shuffle:
                client_cmd.append("--shuffle")

            print("CLIENT:", " ".join(client_cmd))

            with open(client_log, "w", encoding="utf-8") as f:
                ret = subprocess.call(client_cmd, stdout=f, stderr=subprocess.STDOUT)
            if ret != 0:
                raise RuntimeError(f"Client failed with exit code {ret}")

            wall_s = time.time() - t_start

            meta = {
                "run_name": name,
                "run_index": idx,
                "target": target,
                "service_cmd": service_cmd,
                "client_cmd": client_cmd,
                "service_overrides": overrides,
                "dataset_path": args.dataset_path,
                "limit": args.limit,
                "client_params": {
                    "k": args.k,
                    "max_tokens": args.max_tokens,
                    "deadline_s": args.deadline_s,
                    "concurrency": args.concurrency,
                    "retries": args.retries,
                    "retry_backoff_s": args.retry_backoff_s,
                    "shuffle": args.shuffle,
                    "seed": args.seed,
                },
                "telemetry": {
                    "enabled": bool(args.enable_telemetry),
                    "path": str(resource_usage_path) if args.enable_telemetry else None,
                },
                "wall_time_s": wall_s,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "outputs": {
                    "service_log": str(service_log),
                    "client_log": str(client_log),
                    "results_jsonl": str(results_path),
                },
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        finally:
            _stop_process_tree(service_proc)
            try:
                service_log_f.close()
            except Exception:
                pass

    print("\n✅ Sweep complete.")


if __name__ == "__main__":
    main()
