#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile_nearest_rank(sorted_vals: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile, p in [0,100]."""
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    k = int(math.ceil((p / 100.0) * n))
    k = max(1, min(n, k))
    return float(sorted_vals[k - 1])


def summarize_sorted(vals_sorted: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": percentile_nearest_rank(vals_sorted, 50),
        "p90": percentile_nearest_rank(vals_sorted, 90),
        "p95": percentile_nearest_rank(vals_sorted, 95),
        "p99": percentile_nearest_rank(vals_sorted, 99),
        "mean": (sum(vals_sorted) / len(vals_sorted)) if vals_sorted else None,
    }


def fmt(x: Optional[float], *, digits: int = 4) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def load_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def max_fields_from_jsonl(path: Path, fields: List[str]) -> Dict[str, Optional[float]]:
    """
    Compute max numeric value for each field in `fields` across a JSONL file.
    Returns {field: max_or_None}. If file doesn't exist, all Nones.
    """
    out: Dict[str, Optional[float]] = {f: None for f in fields}
    if not path.exists():
        return out

    for rec in iter_jsonl(path):
        for f in fields:
            v = rec.get(f)
            if isinstance(v, (int, float)):
                fv = float(v)
                cur = out[f]
                out[f] = fv if cur is None else max(cur, fv)
    return out


def collect_results_stats(results_path: Path, *, skip_first_n: int) -> Dict[str, Any]:
    ok_lat: List[float] = []
    ok_e2e: List[float] = []
    ok_queue: List[float] = []
    stage_vals: Dict[str, List[float]] = {}

    total = 0
    err_count = 0

    it = iter_jsonl(results_path)

    # skip warmup records regardless of success/failure
    for _ in range(max(0, int(skip_first_n))):
        try:
            next(it)
        except StopIteration:
            break

    for rec in it:
        total += 1
        if rec.get("error"):
            err_count += 1
            continue

        lat = rec.get("client_latency_s")
        if isinstance(lat, (int, float)):
            ok_lat.append(float(lat))

        timings = ((rec.get("trace") or {}).get("timings_s")) or {}

        e2e = timings.get("e2e_s")
        if isinstance(e2e, (int, float)):
            ok_e2e.append(float(e2e))

        q = timings.get("queue_s")
        if isinstance(q, (int, float)):
            ok_queue.append(float(q))

        for k, v in timings.items():
            if isinstance(v, (int, float)):
                stage_vals.setdefault(k, []).append(float(v))

    ok = total - err_count
    err_rate = (err_count / total) if total else 0.0

    ok_lat.sort()
    ok_e2e.sort()
    ok_queue.sort()
    for k in stage_vals:
        stage_vals[k].sort()

    return {
        "total": total,
        "ok": ok,
        "errors": err_count,
        "err_rate": err_rate,
        "lat": ok_lat,
        "e2e": ok_e2e,
        "queue": ok_queue,
        "stages": stage_vals,
    }


def collect_resource_peaks(resource_path: Path) -> Dict[str, Optional[float]]:
    fields = [
        "proc_rss_gb",
        "gpu_mem_used_gb",
        "gpu_util_percent",
        "system_cpu_percent",
        "system_mem_percent",
        "system_mem_used_gb",
    ]
    maxes = max_fields_from_jsonl(resource_path, fields)
    return {
        "proc_rss_gb_peak": maxes["proc_rss_gb"],
        "gpu_mem_used_gb_peak": maxes["gpu_mem_used_gb"],
        "gpu_util_percent_peak": maxes["gpu_util_percent"],
        "system_cpu_percent_peak": maxes["system_cpu_percent"],
        "system_mem_percent_peak": maxes["system_mem_percent"],
        "system_mem_used_gb_peak": maxes["system_mem_used_gb"],
    }


def collect_run(
    run_dir: Path,
    *,
    results_name: str,
    resource_name: str,
    skip_first_n: int,
) -> Optional[Dict[str, Any]]:
    results_path = run_dir / results_name
    if not results_path.exists():
        return None

    meta = load_meta(run_dir)
    wall_time_s = meta.get("wall_time_s")

    stats = collect_results_stats(results_path, skip_first_n=skip_first_n)
    ok = stats["ok"]
    throughput = (ok / wall_time_s) if (isinstance(wall_time_s, (int, float)) and wall_time_s > 0) else None

    peaks = collect_resource_peaks(run_dir / resource_name)

    return {
        "run_dir": str(run_dir),
        "run_name": meta.get("run_name", run_dir.name),
        "target": meta.get("target"),
        "wall_time_s": wall_time_s,
        "throughput_rps": throughput,
        "meta": meta,
        **stats,
        **peaks,
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "run",
        "ok/total",
        "err%",
        "wall_s",
        "rps(ok)",
        "lat_p50",
        "lat_p95",
        "lat_p99",
        "e2e_p50",
        "e2e_p95",
        "queue_p50",
        "queue_p95",
        "rss_peak_gb",
        "gpu_mem_peak_gb",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        lat = summarize_sorted(r["lat"])
        e2e = summarize_sorted(r["e2e"])
        q = summarize_sorted(r["queue"])

        print(
            "| "
            + " | ".join(
                [
                    str(r["run_name"]),
                    f"{r['ok']}/{r['total']}",
                    f"{(100.0 * r['err_rate']):.2f}",
                    fmt(r["wall_time_s"], digits=2),
                    fmt(r["throughput_rps"], digits=3),
                    fmt(lat["p50"]),
                    fmt(lat["p95"]),
                    fmt(lat["p99"]),
                    fmt(e2e["p50"]),
                    fmt(e2e["p95"]),
                    fmt(q["p50"]),
                    fmt(q["p95"]),
                    fmt(r.get("proc_rss_gb_peak"), digits=3),
                    fmt(r.get("gpu_mem_used_gb_peak"), digits=3),
                ]
            )
            + " |"
        )


def print_stage_breakdown(best: Dict[str, Any], stage_keys: List[str]) -> None:
    print("\nStage breakdown (mean / p50 / p95) for:", best["run_name"])
    headers = ["stage", "mean_s", "p50_s", "p95_s"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for k in stage_keys:
        vals = best["stages"].get(k, [])
        s = summarize_sorted(vals)
        print(f"| {k} | {fmt(s['mean'])} | {fmt(s['p50'])} | {fmt(s['p95'])} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/memory_hnsw", help="Directory containing run subdirectories")
    ap.add_argument("--results_name", default="results.jsonl")
    ap.add_argument("--resource_name", default="resource_usage.jsonl")
    ap.add_argument("--skip_first_n", type=int, default=1, help="Skip first N records in each run as warmup")

    ap.add_argument(
        "--sort_by",
        default="lat_p95",
        choices=["lat_p50", "lat_p95", "lat_p99", "e2e_p95", "rps", "err_rate"],
    )
    ap.add_argument("--show_stages", action="store_true")
    ap.add_argument(
        "--stage_keys",
        default="queue_s,ann_s,docstore_s,decode_s,embed_s,prompt_s",
    )
    args = ap.parse_args()

    csv_out = str(Path(args.runs_dir) / "summary.csv")

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    runs: List[Dict[str, Any]] = []
    for rd in run_dirs:
        r = collect_run(
            rd,
            results_name=args.results_name,
            resource_name=args.resource_name,
            skip_first_n=args.skip_first_n,
        )
        if r is not None:
            runs.append(r)

    if not runs:
        print(f"No runs found under {runs_dir} with {args.results_name}")
        return

    # Union of stage keys across runs (for CSV columns)
    all_stage_keys = sorted({k for r in runs for k in r["stages"].keys()})

    def sort_key(r: Dict[str, Any]) -> float:
        lat = summarize_sorted(r["lat"])
        e2e = summarize_sorted(r["e2e"])

        if args.sort_by == "lat_p50":
            return lat["p50"] if lat["p50"] is not None else float("inf")
        if args.sort_by == "lat_p95":
            return lat["p95"] if lat["p95"] is not None else float("inf")
        if args.sort_by == "lat_p99":
            return lat["p99"] if lat["p99"] is not None else float("inf")
        if args.sort_by == "e2e_p95":
            return e2e["p95"] if e2e["p95"] is not None else float("inf")
        if args.sort_by == "rps":
            return r["throughput_rps"] if r["throughput_rps"] is not None else 0.0
        if args.sort_by == "err_rate":
            return r["err_rate"]
        return float("inf")

    runs.sort(key=sort_key, reverse=(args.sort_by == "rps"))

    print_table(runs)

    out = Path(csv_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    resource_headers = [
        "proc_rss_gb_peak",
        # "gpu_mem_used_gb_peak",
        # "gpu_util_percent_peak",
        # "system_cpu_percent_peak",
        # "system_mem_percent_peak",
        "system_mem_used_gb_peak",
    ]
    stage_headers = [f"{k}_mean_s" for k in all_stage_keys]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_name",
                "run_dir",
                "ok",
                "total",
                "err_rate",
                "wall_time_s",
                "throughput_rps",
                "lat_p50",
                "lat_p95",
                "lat_p99",
                "e2e_p50",
                "e2e_p95",
                "e2e_p99",
                "queue_p50",
                "queue_p95",
                "queue_p99",
            ]
            + resource_headers
            + stage_headers
        )

        for r in runs:
            lat = summarize_sorted(r["lat"])
            e2e = summarize_sorted(r["e2e"])
            q = summarize_sorted(r["queue"])

            stage_means: List[Optional[float]] = []
            for k in all_stage_keys:
                vals = r["stages"].get(k, [])
                stage_means.append((sum(vals) / len(vals)) if vals else None)

            w.writerow(
                [
                    r["run_name"],
                    r["run_dir"],
                    r["ok"],
                    r["total"],
                    r["err_rate"],
                    r["wall_time_s"],
                    r["throughput_rps"],
                    lat["p50"],
                    lat["p95"],
                    lat["p99"],
                    e2e["p50"],
                    e2e["p95"],
                    e2e["p99"],
                    q["p50"],
                    q["p95"],
                    q["p99"],
                    r.get("proc_rss_gb_peak"),
                    r.get("gpu_mem_used_gb_peak"),
                    r.get("gpu_util_percent_peak"),
                    r.get("system_cpu_percent_peak"),
                    r.get("system_mem_percent_peak"),
                    r.get("system_mem_used_gb_peak"),
                ]
                + stage_means
            )

    print(f"\nWrote CSV: {out}")

    if args.show_stages:
        best = runs[0]
        stage_keys = [s.strip() for s in args.stage_keys.split(",") if s.strip()]
        print_stage_breakdown(best, stage_keys)


if __name__ == "__main__":
    main()
