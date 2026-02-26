import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PY = sys.executable
SCRIPT = "data/indexes/wiki-dpr/build_wikidpr_fiass_index.py"

ROOT_OUT_DIR = Path("artifacts/wiki-dpr/faiss_wiki_dpr")
MANIFEST_PATH = ROOT_OUT_DIR / "manifest.csv"

DATASET_NAME = "psgs_w100.nq.no_index"
# DATASET_NAME = "psgs_w100.multiset.no_index"

COMMON = [
    "--no-streaming",
    "--dataset_name", DATASET_NAME,
    "--snippet_chars", "0",
    # you probably want meta small for full runs; keep 0 text snippets
    # "--store_text", "false"  # (only if your builder supports a flag; currently it doesn't)
]

# You want exactly two sizes: 100k and full corpus
SIZES = [
    ("100k", 100_000),
    ("all", None),  # full split (requires builder script with --all_vectors)
]

JOBS: list[tuple[str, list[str]]] = []


# ---------------- HNSW params ----------------
def hnsw_params(n: Optional[int]) -> tuple[str, str]:
    # for "all", assume big
    if n is None:
        return "32768", "320"
    if n <= 100_000:
        return "16384", "64"
    if n <= 500_000:
        return "16384", "96"
    if n <= 1_000_000:
        return "16384", "128"
    if n <= 2_500_000:
        return "32768", "160"
    if n <= 5_000_000:
        return "32768", "192"
    if n <= 10_000_000:
        return "32768", "256"
    return "32768", "320"


# ---------------- IVF params ----------------
def ivf_params(n: Optional[int]) -> tuple[str, str, str, str]:
    """
    Returns (n_lists, train_size, nprobe, batch_size) as strings for CLI args.

    Heuristics (FAISS IVF_IP):
      - n_lists (nlist) ~ 4*sqrt(n), clamped [1024, 65536], snapped to power-of-two
      - train_size ~ 100 samples per list, capped to [100k, 2M] and <= n (if n known)
      - nprobe ~ 0.5% of nlist, clamped [16, 256]
      - batch_size: build/add batching
    """
    if n is None:
        # "all" — pick conservative big-corpus defaults (tweak if you want) 
        nlist = 4580
        train_size = 2_000_000
        nprobe = 256
        batch = 32768
        return str(nlist), str(train_size), str(nprobe), str(batch)

    target = 4.0 * math.sqrt(n)
    target = max(1024.0, min(65536.0, target))
    nlist = int(2 ** round(math.log2(target)))
    nlist = max(1024, min(65536, nlist))

    train_size = min(n, max(100_000, min(2_000_000, 100 * nlist)))
    nprobe = int(max(16, min(256, round(0.005 * nlist))))
    batch = 16384 if n <= 1_000_000 else 32768

    return str(nlist), str(train_size), str(nprobe), str(batch)


# ---------------- Build job list: 2 per index type ----------------
for tag, n in SIZES:
    # FLAT
    flat_args = ["--index_type", "flat_ip"]
    if n is None:
        flat_args += ["--all_vectors"]
    else:
        flat_args += ["--n_vectors", str(n)]
    JOBS.append((f"flat_{tag}", flat_args))

    # HNSW
    batch, ef_search = hnsw_params(n)
    hnsw_args = [
        "--index_type", "hnsw_ip",
        "--batch_size", batch,
        "--n_neighbors", "32",
        "--ef_construction", "200",
        "--ef_search", ef_search,
    ]
    if n is None:
        hnsw_args += ["--all_vectors"]
    else:
        hnsw_args += ["--n_vectors", str(n)]
    JOBS.append((f"hnsw_{tag}", hnsw_args))

    # IVF
    n_lists, train_size, nprobe, batch = ivf_params(n)
    ivf_args = [
        "--index_type", "ivf_ip",
        "--batch_size", batch,
        "--n_lists", n_lists,
        "--train_size", train_size,
        "--nprobe", nprobe,
    ]
    if n is None:
        ivf_args += ["--all_vectors"]
    else:
        ivf_args += ["--n_vectors", str(n)]
    JOBS.append((f"ivf_{tag}", ivf_args))


print("Planned jobs:")
for name, args in JOBS:
    print(f"  {name}: {' '.join(args)}")


# ---------------- metrics helpers ----------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bytes_human(n: Optional[int]) -> str:
    if n is None:
        return ""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} B"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except FileNotFoundError:
                pass
    return total


def try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


psutil = try_import_psutil()


@dataclass
class RunResult:
    returncode: int
    elapsed_sec: float
    peak_rss_bytes: Optional[int]


def run_and_track(cmd: list[str]) -> RunResult:
    t0 = time.perf_counter()

    if psutil is None:
        p = subprocess.run(cmd)
        return RunResult(p.returncode, time.perf_counter() - t0, None)

    proc = subprocess.Popen(cmd)
    peak = 0

    try:
        ps_proc = psutil.Process(proc.pid)
    except Exception:
        proc.wait()
        return RunResult(proc.returncode or 0, time.perf_counter() - t0, None)

    while True:
        if proc.poll() is not None:
            break
        try:
            rss = 0
            procs = [ps_proc] + ps_proc.children(recursive=True)
            for pr in procs:
                try:
                    rss += pr.memory_info().rss
                except Exception:
                    pass
            peak = max(peak, rss)
        except Exception:
            pass
        time.sleep(0.5)

    proc.wait()

    # final sample
    try:
        rss = 0
        procs = [ps_proc] + ps_proc.children(recursive=True)
        for pr in procs:
            try:
                rss += pr.memory_info().rss
            except Exception:
                pass
        peak = max(peak, rss)
    except Exception:
        pass

    return RunResult(proc.returncode or 0, time.perf_counter() - t0, peak if peak > 0 else None)


# ---------------- config extraction ----------------
CONFIG_KEYS = [
    "index_type",
    "n_vectors",
    "all_vectors",
    "batch_size",
    # IVF-specific:
    "n_lists",
    "nprobe",
    "train_size",
    # HNSW-specific:
    "n_neighbors",
    "ef_construction",
    "ef_search",
]


def parse_config(extra_args: list[str]) -> dict:
    """
    Parses args like:
      --index_type ivf_ip --n_vectors 1000000 --n_lists 4096 ...
    into a dict {index_type: "...", n_vectors: "...", ...}
    """
    cfg: dict[str, str] = {}
    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if tok.startswith("--"):
            key = tok[2:]
            val: str = "true"
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                val = extra_args[i + 1]
                i += 1
            cfg[key] = val
        i += 1
    return cfg


# ---------------- manifest writing ----------------
MANIFEST_FIELDS = [
    "job_name",
    "dataset_name",
    "out_dir",
    *CONFIG_KEYS,
    "config_json",
    "cmd",
    "start_time_utc",
    "end_time_utc",
    "elapsed_sec",
    "returncode",
    "disk_bytes",
    "disk_human",
    "peak_rss_bytes",
    "peak_rss_human",
]


def append_manifest_row(row: dict):
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = MANIFEST_PATH.exists()

    with MANIFEST_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not file_exists:
            writer.writeheader()
        safe_row = {k: row.get(k, "") for k in MANIFEST_FIELDS}
        writer.writerow(safe_row)


def run_job(job_name: str, extra_args: list[str]):
    job_out_dir = ROOT_OUT_DIR / job_name
    job_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY,
        SCRIPT,
        *COMMON,
        "--out_dir",
        str(job_out_dir),
        *extra_args,
    ]

    print("\n== Running:", job_name)
    print("   ", shlex.join(cmd))

    cfg = parse_config(extra_args)

    start = utc_now_iso()
    result = run_and_track(cmd)
    end = utc_now_iso()

    disk = dir_size_bytes(job_out_dir) if job_out_dir.exists() else 0

    row = {
        "job_name": job_name,
        "dataset_name": DATASET_NAME,
        "out_dir": str(job_out_dir),
        **{k: cfg.get(k, "") for k in CONFIG_KEYS},
        "config_json": json.dumps(cfg, sort_keys=True),
        "cmd": shlex.join(cmd),
        "start_time_utc": start,
        "end_time_utc": end,
        "elapsed_sec": f"{result.elapsed_sec:.3f}",
        "returncode": str(result.returncode),
        "disk_bytes": str(disk),
        "disk_human": bytes_human(disk),
        "peak_rss_bytes": "" if result.peak_rss_bytes is None else str(result.peak_rss_bytes),
        "peak_rss_human": bytes_human(result.peak_rss_bytes),
    }

    append_manifest_row(row)

    if result.returncode != 0:
        print(f"!! Job failed: {job_name} (returncode={result.returncode})")


def main():
    if not Path(SCRIPT).exists():
        raise FileNotFoundError(f"Can't find {SCRIPT} in {Path.cwd()}")

    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, args in JOBS:
        run_job(name, args)

    print("\nAll jobs completed.")
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()