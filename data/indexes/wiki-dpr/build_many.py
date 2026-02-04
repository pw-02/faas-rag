import subprocess
import sys
from pathlib import Path

PY = sys.executable  # uses the same python env you're running this with
SCRIPT = "build_wiki_dpr_index.py"

OUT_DIR = "faiss_wiki_dpr"
DATASET_NAME = "psgs_w100.nq.no_index"
# DATASET_NAME = "psgs_w100.multiset.no_index"

COMMON = [
    "--no-streaming",
    "--dataset_name", DATASET_NAME,
    "--out_dir", OUT_DIR,
    "--snippet_chars", "0",
]

JOBS = [
    # name, args...
    ("flat_10k",  ["--index_type", "flat_ip", "--n_vectors", "10000"]),
    ("flat_100k", ["--index_type", "flat_ip", "--n_vectors", "100000"]),
    ("hnsw_100k", ["--index_type", "hnsw_ip", "--n_vectors", "100000",
                  "--n_neighbors", "32", "--ef_construction", "200", "--ef_search", "64"]),
    ("hnsw_1m",   ["--index_type", "hnsw_ip", "--n_vectors", "1000000",
                  "--batch_size", "16384",
                  "--n_neighbors", "32", "--ef_construction", "200", "--ef_search", "128"]),
    ("ivf_1m",    ["--index_type", "ivf_ip", "--n_vectors", "1000000",
                  "--batch_size", "16384",
                  "--n_lists", "4096", "--train_size", "100000", "--nprobe", "32"]),
    # Uncomment when ready:
    # ("ivf_5m",    ["--index_type", "ivf_ip", "--n_vectors", "5000000",
    #               "--batch_size", "32768",
    #               "--n_lists", "8192", "--train_size", "200000", "--nprobe", "64"]),
]

def run_job(name: str, extra_args: list[str]):
    cmd = [PY, SCRIPT, *COMMON, *extra_args]
    print("\n== Running:", name)
    print("   ", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    # basic sanity checks
    if not Path(SCRIPT).exists():
        raise FileNotFoundError(f"Can't find {SCRIPT} in {Path.cwd()}")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    for name, args in JOBS:
        run_job(name, args)

    print("\nAll jobs completed.")

if __name__ == "__main__":
    main()
