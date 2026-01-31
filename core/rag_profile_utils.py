from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def save_batch_results_csv(results: list[dict[str, Any]], path: str) -> None:
    """Save batch-level results (one row per batch)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in results:
        timings = r.get("timings", {}) or {}

        batch_size = int(r.get("batch_size", 0) or 0)
        total_s = float(timings.get("total_s", 0.0) or 0.0)

        rows.append({
            "batch_start_idx": int(r.get("batch_start", 0) or 0),
            "batch_size": batch_size,
            "top_k": int(r.get("top_k", 0) or 0),
            "max_context_docs": int(r.get("max_context_docs", 0) or 0),

            "embed_time_s": float(timings.get("embed_s", 0.0) or 0.0),
            "faiss_search_time_s": float(timings.get("faiss_s", 0.0) or 0.0),
            "docstore_fetch_time_s": float(timings.get("docstore_s", 0.0) or 0.0),
            "prompt_build_time_s": float(timings.get("prompt_s", 0.0) or 0.0),
            "generation_time_s": float(timings.get("generate_s", 0.0) or 0.0),
            "total_time_s": total_s,

            "throughput_qps": (batch_size / total_s) if total_s > 0 else 0.0,
            "avg_latency_per_query_s": (total_s / batch_size) if batch_size > 0 else 0.0,
        })

    if not rows:
        return

    with path_obj.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_summary_from_csvs(csv_paths: list[str], summary_csv_output_path: str) -> pd.DataFrame:
    """Return a summary dataframe and write it to disk."""
    summary_rows: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        p = Path(csv_path)
        df = pd.read_csv(p)
        if df.empty:
            continue

        total_queries = int(df["batch_size"].sum())
        total_time = float(df["total_time_s"].sum())
        overall_qps = (total_queries / total_time) if total_time > 0 else 0.0

        summary_rows.append({
            "index": p.stem,
            "batch_size": int(df["batch_size"].iloc[0]),
            "num_batches": int(len(df)),
            "total_queries": total_queries,

            "avg_total_time_s": float(df["total_time_s"].mean()),
            "avg_embed_time_s": float(df["embed_time_s"].mean()),
            "avg_faiss_search_time_s": float(df["faiss_search_time_s"].mean()),
            "avg_docstore_fetch_time_s": float(df["docstore_fetch_time_s"].mean()),
            "avg_prompt_build_time_s": float(df["prompt_build_time_s"].mean()),
            "avg_generation_time_s": float(df["generation_time_s"].mean()),

            "p95_total_time_s": float(df["total_time_s"].quantile(0.95)),

            "avg_batch_qps": float(df["throughput_qps"].mean()),
            "overall_qps": overall_qps,
            "avg_latency_per_query_s": float(df["avg_latency_per_query_s"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    out = Path(summary_csv_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)

    return summary_df


def load_queries_from_file(
    path: str,
    *,
    column: str = "question",
    batch_size: int = 1,
    max_batches: Optional[int] = None,
) -> list[str]:
    df = pd.read_csv(path)
    queries = df[column].dropna().astype(str)

    if max_batches is not None:
        max_queries = max_batches * batch_size
        queries = queries.head(max_queries)

    return queries.tolist()
