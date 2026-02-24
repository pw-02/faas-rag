#!/usr/bin/env python3#
import argparse
import json
import os
import re
from pathlib import Path

def slug(s: str) -> str:
    """Make a safe folder name."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/datasets/mmlu/mmlu/mmlu_train.jsonl", help="Path to input JSONL file (one JSON object per line).")
    ap.add_argument("--out_dir", default="data/datasets/mmlu/subjects/", help="Output root directory.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Keep files open for speed (optional), but simple + safe.
    writers = {}

    def get_writer(subject_folder: str):
        if subject_folder not in writers:
            folder = out_root / subject_folder
            folder.mkdir(parents=True, exist_ok=True)
            f = open(folder / "train.jsonl", "a", encoding="utf-8")
            writers[subject_folder] = f
        return writers[subject_folder]

    total = 0
    bad = 0

    with open(in_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue

            subject = None
            md = obj.get("metadata")
            if isinstance(md, dict):
                subject = md.get("subject")
            
            if isinstance(subject, str) and subject.strip():
                subject_folder = slug(subject)
                # Write the ORIGINAL line back out (preserves exactly what was in the file)
                w = get_writer(subject_folder)
                w.write(line + "\n")
                total += 1
            else:
                #skip bad line if no subject and no unknown_subject provided
                bad += 1
                continue

      

    # Close writers
    for f in writers.values():
        f.close()

    print(f"Done. Wrote {total} lines into {len(writers)} subject folders under: {out_root}")
    if bad:
        print(f"Skipped {bad} bad JSON lines.")


if __name__ == "__main__":
    main()