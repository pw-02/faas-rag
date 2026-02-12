#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# -------------------------
# JSONL helpers
# -------------------------

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


# -------------------------
# Meta + resource peaks
# -------------------------

def load_meta(experiment_dir: Path) -> Dict[str, Any]:
    meta_path = experiment_dir / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def max_fields_from_jsonl(path: Path, fields: List[str]) -> Dict[str, Optional[float]]:
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


def collect_resource_peaks(resource_path: Optional[Path]) -> Dict[str, Optional[float]]:
    fields = [
        "proc_rss_gb",
        "gpu_mem_used_gb",
        "gpu_util_percent",
        "system_cpu_percent",
        "system_mem_percent",
        "system_mem_used_gb",
    ]
    if resource_path is None or not resource_path.exists():
        return {
            "proc_rss_gb_peak": None,
            "gpu_mem_used_gb_peak": None,
            "gpu_util_percent_peak": None,
            "system_cpu_percent_peak": None,
            "system_mem_percent_peak": None,
            "system_mem_used_gb_peak": None,
        }
    maxes = max_fields_from_jsonl(resource_path, fields)
    return {
        "proc_rss_gb_peak": maxes["proc_rss_gb"],
        "gpu_mem_used_gb_peak": maxes["gpu_mem_used_gb"],
        "gpu_util_percent_peak": maxes["gpu_util_percent"],
        "system_cpu_percent_peak": maxes["system_cpu_percent"],
        "system_mem_percent_peak": maxes["system_mem_percent"],
        "system_mem_used_gb_peak": maxes["system_mem_used_gb"],
    }


# -------------------------
# Results stats (latency, stages, cache, tokens)
# -------------------------

def collect_results_stats(results_path: Path, *, skip_first_n: int) -> Dict[str, Any]:
    ok_lat: List[float] = []
    ok_e2e: List[float] = []
    ok_queue: List[float] = []
    stage_vals: Dict[str, List[float]] = {}

    total = 0
    err_count = 0

    cache_hits_total = 0
    cache_misses_total = 0
    prompt_tokens_total = 0
    completion_tokens_total = 0
    total_tokens_total = 0

    index_vector_count: Optional[int] = None

    it = iter_jsonl(results_path)

    # skip warmup regardless of success/failure
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

        trace = rec.get("trace") or {}
        timings = trace.get("timings_s") or {}

        if index_vector_count is None:
            ivc = trace.get("index_vector_count")
            if isinstance(ivc, int):
                index_vector_count = ivc

        e2e = timings.get("e2e_s")
        if isinstance(e2e, (int, float)):
            ok_e2e.append(float(e2e))

        q = timings.get("queue_s")
        if isinstance(q, (int, float)):
            ok_queue.append(float(q))

        for k, v in timings.items():
            if isinstance(v, (int, float)):
                stage_vals.setdefault(k, []).append(float(v))

        ch = trace.get("cache_hits")
        cm = trace.get("cache_misses")
        if isinstance(ch, int):
            cache_hits_total += ch
        if isinstance(cm, int):
            cache_misses_total += cm

        pt = trace.get("prompt_tokens")
        ct = trace.get("completion_tokens")
        tt = trace.get("total_tokens")
        if isinstance(pt, int):
            prompt_tokens_total += pt
        if isinstance(ct, int):
            completion_tokens_total += ct
        if isinstance(tt, int):
            total_tokens_total += tt

    ok = total - err_count
    err_rate = (err_count / total) if total else 0.0

    ok_lat.sort()
    ok_e2e.sort()
    ok_queue.sort()
    for k in stage_vals:
        stage_vals[k].sort()

    cache_total = cache_hits_total + cache_misses_total
    cache_hit_rate = (cache_hits_total / cache_total) if cache_total else None

    prompt_tokens_avg = (prompt_tokens_total / ok) if ok else None
    completion_tokens_avg = (completion_tokens_total / ok) if ok else None
    total_tokens_avg = (total_tokens_total / ok) if ok else None

    return {
        "index_vector_count": index_vector_count,
        "total": total,
        "ok": ok,
        "errors": err_count,
        "err_rate": err_rate,
        "lat": ok_lat,
        "e2e": ok_e2e,
        "queue": ok_queue,
        "stages": stage_vals,
        "cache_hits_total": cache_hits_total,
        "cache_misses_total": cache_misses_total,
        "cache_hit_rate": cache_hit_rate,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "prompt_tokens_avg": prompt_tokens_avg,
        "completion_tokens_avg": completion_tokens_avg,
        "total_tokens_avg": total_tokens_avg,
    }


# -------------------------
# Collect a single experiment
# -------------------------

def collect_experiment(
    experiment_dir: Path,
    *,
    results_name: str,
    resource_name: str,
    skip_first_n: int,
) -> Optional[Dict[str, Any]]:
    results_path = experiment_dir / results_name
    if not results_path.exists():
        return None

    meta = load_meta(experiment_dir)
    wall_time_s = meta.get("wall_time_s")

    stats = collect_results_stats(results_path, skip_first_n=skip_first_n)
    ok = stats["ok"]
    throughput = (ok / wall_time_s) if (isinstance(wall_time_s, (int, float)) and wall_time_s > 0) else None

    resource_path = experiment_dir / resource_name
    peaks = collect_resource_peaks(resource_path if resource_path.exists() else None)

    return {
        "experiment_dir": str(experiment_dir),
        "experiment_name": meta.get("run_name", experiment_dir.name),
        "wall_time_s": wall_time_s,
        "throughput_rps": throughput,
        "meta": meta,
        **stats,
        **peaks,
    }


# -------------------------
# CSV writing (ONE summary.csv per experiment folder)
# -------------------------

def write_experiment_summary_csv(
    experiment_dir: Path,
    exp: Dict[str, Any],
    *,
    out_name: str,
    stage_order_preference: List[str],
) -> Path:
    out_path = experiment_dir / f"{exp['experiment_name']}_{out_name}"

    # Keep preferred stage order first, then append any extras (stable)
    seen = set((exp.get("stages") or {}).keys())
    stage_keys = [k for k in stage_order_preference if k in seen] + sorted(seen - set(stage_order_preference))
    stage_headers = [f"{k}_mean_s" for k in stage_keys]

    token_cache_headers = [
        "cache_hits_total",
        "cache_misses_total",
        "cache_hit_rate",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
        "prompt_tokens_avg",
        "completion_tokens_avg",
        "total_tokens_avg",
    ]

    resource_headers = [
        "proc_rss_gb_peak",
        "gpu_mem_used_gb_peak",
        "gpu_util_percent_peak",
        "system_cpu_percent_peak",
        "system_mem_percent_peak",
        "system_mem_used_gb_peak",
    ]

    # percentile summaries
    lat = summarize_sorted(exp["lat"])
    e2e = summarize_sorted(exp["e2e"])
    q = summarize_sorted(exp["queue"])

    # stage means
    stage_means: List[Optional[float]] = []
    for k in stage_keys:
        vals = exp["stages"].get(k, [])
        stage_means.append((sum(vals) / len(vals)) if vals else None)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment_name",
                "experiment_dir",
                "index_vector_count",
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
            + token_cache_headers
            + resource_headers
            + stage_headers
        )

        w.writerow(
            [
                exp["experiment_name"],
                exp["experiment_dir"],
                exp.get("index_vector_count"),
                exp["ok"],
                exp["total"],
                exp["err_rate"],
                exp.get("wall_time_s"),
                exp.get("throughput_rps"),
                lat["p50"],
                lat["p95"],
                lat["p99"],
                e2e["p50"],
                e2e["p95"],
                e2e["p99"],
                q["p50"],
                q["p95"],
                q["p99"],
                # tokens/cache
                exp.get("cache_hits_total"),
                exp.get("cache_misses_total"),
                exp.get("cache_hit_rate"),
                exp.get("prompt_tokens_total"),
                exp.get("completion_tokens_total"),
                exp.get("total_tokens_total"),
                exp.get("prompt_tokens_avg"),
                exp.get("completion_tokens_avg"),
                exp.get("total_tokens_avg"),
                # peaks
                exp.get("proc_rss_gb_peak"),
                exp.get("gpu_mem_used_gb_peak"),
                exp.get("gpu_util_percent_peak"),
                exp.get("system_cpu_percent_peak"),
                exp.get("system_mem_percent_peak"),
                exp.get("system_mem_used_gb_peak"),
            ]
            + stage_means
        )

    return out_path


# -------------------------
# Optional: print an overall table (for convenience)
# -------------------------

def print_overall_table(exps: List[Dict[str, Any]]) -> None:
    headers = [
        "experiment",
        "ok/total",
        "err%",
        "wall_s",
        "rps(ok)",
        "lat_p95",
        "e2e_p95",
        "rss_peak_gb",
        "gpu_mem_peak_gb",
        "cache_hit_rate",
        "tok_total_avg",
    ]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for e in exps:
        lat = summarize_sorted(e["lat"])
        e2e = summarize_sorted(e["e2e"])
        print(
            "| "
            + " | ".join(
                [
                    str(e["experiment_name"]),
                    f"{e['ok']}/{e['total']}",
                    f"{(100.0 * e['err_rate']):.2f}",
                    fmt(e.get("wall_time_s"), digits=2),
                    fmt(e.get("throughput_rps"), digits=3),
                    fmt(lat["p95"]),
                    fmt(e2e["p95"]),
                    fmt(e.get("proc_rss_gb_peak"), digits=3),
                    fmt(e.get("gpu_mem_used_gb_peak"), digits=3),
                    fmt(e.get("cache_hit_rate"), digits=3),
                    fmt(e.get("total_tokens_avg"), digits=2),
                ]
            )
            + " |"
        )


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs", help="Root dir containing experiment folders (nested OK)")
    ap.add_argument("--results_name", default="results.jsonl")
    ap.add_argument("--resource_name", default="resource_usage.jsonl")
    ap.add_argument("--summary_name", default="summary.csv", help="Filename to write inside each experiment folder")
    ap.add_argument("--skip_first_n", type=int, default=1, help="Skip first N records in each run as warmup")
    ap.add_argument("--stage_keys", default="queue_s,ann_s,docstore_s,decode_s,embed_s,prompt_s")
    ap.add_argument("--print_table", action="store_true", help="Print an overall markdown table across experiments")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)

    # Find all results.jsonl anywhere under runs_dir; each parent folder is an experiment dir
    results_files = sorted(runs_dir.rglob(args.results_name))
    if not results_files:
        print(f"No {args.results_name} found anywhere under {runs_dir}")
        return

    stage_pref = [s.strip() for s in args.stage_keys.split(",") if s.strip()]

    experiments: List[Dict[str, Any]] = []
    written = 0

    # De-dupe experiment dirs if multiple results.jsonl somehow exist in same folder
    exp_dirs = sorted({p.parent for p in results_files})

    for exp_dir in exp_dirs:
        exp = collect_experiment(
            exp_dir,
            results_name=args.results_name,
            resource_name=args.resource_name,
            skip_first_n=args.skip_first_n,
        )
        if exp is None:
            continue

        #chacnge sumamry name to include experiment name


        out_path = write_experiment_summary_csv(
            exp_dir,
            exp,
            out_name=args.summary_name,
            stage_order_preference=stage_pref,
        )
        written += 1
        experiments.append(exp)

        print(f"Wrote {out_path}")

    if not written:
        print("No experiments summarized (no folders with results.jsonl found).")
        return

    if args.print_table:
        print_overall_table(experiments)


if __name__ == "__main__":
    main()
