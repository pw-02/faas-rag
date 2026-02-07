#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
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


def _fmt(x: Optional[float], *, digits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _collect_run(
    run_dir: Path,
    *,
    results_name: str = "results.jsonl",
    skip_first_n: int = 0,
) -> Optional[Dict[str, Any]]:
    results_path = run_dir / results_name
    if not results_path.exists():
        return None

    meta = _load_meta(run_dir)
    wall_time_s = meta.get("wall_time_s")  # may be None

    ok_lat: List[float] = []
    ok_e2e: List[float] = []
    ok_queue: List[float] = []
    err_count = 0
    total = 0

    stage_vals: Dict[str, List[float]] = {}

    # Skip warmup records (regardless of error/success)
    it = _iter_jsonl(results_path)
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
    throughput = (ok / wall_time_s) if (wall_time_s and wall_time_s > 0) else None

    ok_lat.sort()
    ok_e2e.sort()
    ok_queue.sort()
    for k in stage_vals:
        stage_vals[k].sort()

    return {
        "run_dir": str(run_dir),
        "run_name": meta.get("run_name", run_dir.name),
        "target": meta.get("target"),
        "wall_time_s": wall_time_s,
        "total": total,
        "ok": ok,
        "errors": err_count,
        "err_rate": err_rate,
        "throughput_rps": throughput,
        "lat": ok_lat,
        "e2e": ok_e2e,
        "queue": ok_queue,
        "stages": stage_vals,
        "meta": meta,
    }


def _summarize_series(vals_sorted: List[float]) -> Dict[str, Optional[float]]:
    return {
        "p50": _percentile(vals_sorted, 50),
        "p90": _percentile(vals_sorted, 90),
        "p95": _percentile(vals_sorted, 95),
        "p99": _percentile(vals_sorted, 99),
        "mean": (sum(vals_sorted) / len(vals_sorted)) if vals_sorted else None,
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
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
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        lat = _summarize_series(r["lat"])
        e2e = _summarize_series(r["e2e"])
        queue = _summarize_series(r["queue"])

        print(
            "| "
            + " | ".join(
                [
                    str(r["run_name"]),
                    f'{r["ok"]}/{r["total"]}',
                    f'{(100.0 * r["err_rate"]):.2f}',
                    _fmt(r["wall_time_s"], digits=2),
                    _fmt(r["throughput_rps"], digits=3),
                    _fmt(lat["p50"]),
                    _fmt(lat["p95"]),
                    _fmt(lat["p99"]),
                    _fmt(e2e["p50"]),
                    _fmt(e2e["p95"]),
                    _fmt(queue["p50"]),
                    _fmt(queue["p95"]),
                ]
            )
            + " |"
        )


def _print_stage_breakdown(best: Dict[str, Any], stage_keys: List[str]) -> None:
    print("\nStage breakdown (mean / p50 / p95) for:", best["run_name"])
    headers = ["stage", "mean_s", "p50_s", "p95_s"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    for k in stage_keys:
        vals = best["stages"].get(k, [])
        s = _summarize_series(vals)
        print(f"| {k} | {_fmt(s['mean'])} | {_fmt(s['p50'])} | {_fmt(s['p95'])} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs/aws_r6a12xlarge")
    ap.add_argument("--results_name", default="results.jsonl")
    ap.add_argument("--skip_first_n", type=int, default=1, help="Skip first N records in each run as warmup")

    ap.add_argument(
        "--sort_by",
        default="lat_p95",
        choices=["lat_p50", "lat_p95", "lat_p99", "e2e_p95", "rps", "err_rate"],
    )
    ap.add_argument("--csv_out", default="runs/aws_r6a12xlarge/summary.csv")
    ap.add_argument("--show_stages", action="store_true")
    ap.add_argument(
        "--stage_keys",
        default="queue_s,e2e_s,pipeline_s,ann_s,docstore_s,decode_s,embed_s,total_s",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()])

    runs: List[Dict[str, Any]] = []
    for rd in run_dirs:
        r = _collect_run(rd, results_name=args.results_name, skip_first_n=args.skip_first_n)
        if r is not None:
            runs.append(r)

    if not runs:
        print(f"No runs found under {runs_dir} with {args.results_name}")
        return

    # Union of stage keys across runs (for CSV columns)
    all_stage_keys = set()
    for r in runs:
        all_stage_keys.update(r["stages"].keys())
    all_stage_keys = sorted(all_stage_keys)

    def key_for(r: Dict[str, Any]) -> float:
        lat = _summarize_series(r["lat"])
        e2e = _summarize_series(r["e2e"])
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

    runs.sort(key=key_for, reverse=(args.sort_by == "rps"))

    _print_table(runs)

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)

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
                + stage_headers
            )

            for r in runs:
                lat = _summarize_series(r["lat"])
                e2e = _summarize_series(r["e2e"])
                q = _summarize_series(r["queue"])

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
                    ]
                    + stage_means
                )

        print(f"\nWrote CSV: {out}")

    if args.show_stages:
        best = runs[0]
        stage_keys = [s.strip() for s in args.stage_keys.split(",") if s.strip()]
        _print_stage_breakdown(best, stage_keys)


if __name__ == "__main__":
    main()
