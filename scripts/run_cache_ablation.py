#!/usr/bin/env python3

import subprocess

INDEXES = [
    "wiki_dpr_ivf_21m",
    "wiki_dpr_hnsw_21m",
    "wiki_dpr_flat_21m",
]

DATSETS = [
    # "mmlu_econometrics",
    # "nq",
    "repeated_question",
]

for dataset in DATSETS:
    print(f"\n=== Running dataset: {dataset} ===")
    for idx in INDEXES:
        print(f"\n=== Running index: {idx} (no cache) ===")

        subprocess.run(
            [
                "python",
                "pwrag/client/eval_dataset.py",
                f"retriever/index={idx}",
                "retriever/cache=none",
                f"save_dir=results/{dataset}/{idx}/none",
            ],
            check=True,
        )

        print(f"\n=== Running index: {idx} (proximity cache) ===")

        subprocess.run(
            [
                "python",
                "pwrag/client/eval_dataset.py",
                f"retriever/index={idx}",
                "retriever/cache=proximity",
                f"save_dir=results/{dataset}/{idx}/proximity",
            ],
            check=True,
        )