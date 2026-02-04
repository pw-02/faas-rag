import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PY = sys.executable
SCRIPT = "build_wikidpr_fiass_index.py"  # <- keep your filename as-is
ROOT_OUT_DIR = Path("faiss_wiki_dpr_meta")
MANIFEST_PATH = ROOT_OUT_DIR / "manifest.csv"
DATASET_NAME = "psgs_w100.nq.no_index"

COMMON = [
    "--no-streaming",
    "--dataset_name", DATASET_NAME,
    "--snippet_chars", "0",
    "--no_index",        # meta-only
    "--store_text",      # include full text in meta jsonl
]

SIZES = [
    ("100k", 100_000),
    ("500k", 500_000),
    ("1m", 1_000_000),
    ("2_5m", 2_500_000),
    ("5m", 5_000_000),
    ("10m", 10_000_000),
    ("21m", 21_000_000),
]

@dataclass
class RunResult:
    tag: str
    n_vectors: int
    out_dir: str
    meta_path: str
    started_at: str
    finished_at: str
    seconds: float
    returncode: int
    error: Optional[str] = None

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def meta_filename(dataset_name: str, n_vectors: int) -> str:
    safe_cfg = dataset_name.replace(".", "_")
    return f"meta_{safe_cfg}_{n_vectors}.jsonl"

def run_one(tag: str, n_vectors: int, root_out: Path) -> RunResult:
    # Option B: per-size subdir (clean + convenient)
    out_dir = root_out / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / meta_filename(DATASET_NAME, n_vectors)

    cmd = [
        PY, SCRIPT,
        *COMMON,
        "--n_vectors", str(n_vectors),
        "--out_dir", str(out_dir),
    ]

    started = utc_now_iso()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False)
        rc = proc.returncode
        err = None
    except Exception as e:
        rc = 999
        err = repr(e)
    seconds = time.time() - t0
    finished = utc_now_iso()

    return RunResult(
        tag=tag,
        n_vectors=n_vectors,
        out_dir=str(out_dir),
        meta_path=str(meta_path),
        started_at=started,
        finished_at=finished,
        seconds=seconds,
        returncode=rc,
        error=err,
    )

def write_manifest(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag", "n_vectors", "out_dir", "meta_path",
                "started_at", "finished_at", "seconds",
                "returncode", "error",
            ],
        )
        if is_new:
            w.writeheader()
        for r in results:
            w.writerow({
                "tag": r.tag,
                "n_vectors": r.n_vectors,
                "out_dir": r.out_dir,
                "meta_path": r.meta_path,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "seconds": f"{r.seconds:.2f}",
                "returncode": r.returncode,
                "error": r.error or "",
            })

def main():
    ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for tag, n in SIZES:
        print(f"\n=== META BUILD {tag} ({n:,}) ===")
        r = run_one(tag, n, ROOT_OUT_DIR)
        results.append(r)

        if r.returncode != 0:
            print(f"[WARN] run failed tag={tag} rc={r.returncode} error={r.error}")
        else:
            print(f"[OK] wrote: {r.meta_path}")

        # write manifest incrementally so you don't lose progress
        write_manifest([r], MANIFEST_PATH)

    print("\nAll done. Manifest at:", MANIFEST_PATH)

if __name__ == "__main__":
    main()
