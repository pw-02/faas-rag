import os
import json
from collections import Counter, defaultdict

file_path = r"C:\Users\pw\Desktop\wiki18_100w.jsonl"

# Tuning knobs
N_PREVIEW = 2         # how many rows to print
MAX_ROWS_SCAN = None    # set to an int (e.g., 200000) to limit scan; None = scan whole file
PRINT_PREVIEW_AS_JSON = True  # pretty print preview rows

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def safe_type(v):
    # keep it readable and stable
    if v is None:
        return "null"
    t = type(v).__name__
    return t

def summarize_value(v):
    """Return (type_name, length_or_none) for common containers/strings."""
    if v is None:
        return ("null", None)
    if isinstance(v, str):
        return ("str", len(v))
    if isinstance(v, (list, tuple, set)):
        return (type(v).__name__, len(v))
    if isinstance(v, dict):
        return ("dict", len(v))
    return (type(v).__name__, None)

def jsonl_inspect(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    file_size = os.path.getsize(path)

    # Stats
    total_rows = 0
    bad_json_rows = 0
    empty_lines = 0

    keys_seen = set()
    key_counts = Counter()              # how often each key appears
    type_counts = defaultdict(Counter)  # key -> Counter(type_name)
    length_stats = defaultdict(lambda: {"count": 0, "min": None, "max": None, "sum": 0})  # for str/list/dict

    preview_raw = []
    preview_parsed = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if MAX_ROWS_SCAN is not None and total_rows >= MAX_ROWS_SCAN:
                break

            line = line.rstrip("\n")
            if not line.strip():
                empty_lines += 1
                continue

            total_rows += 1

            # Save preview
            if len(preview_raw) < N_PREVIEW:
                preview_raw.append(line)

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_json_rows += 1
                continue

            if len(preview_parsed) < N_PREVIEW:
                preview_parsed.append(obj)

            if isinstance(obj, dict):
                for k, v in obj.items():
                    keys_seen.add(k)
                    key_counts[k] += 1

                    tname, vlen = summarize_value(v)
                    type_counts[k][tname] += 1

                    if vlen is not None:
                        st = length_stats[k]
                        st["count"] += 1
                        st["sum"] += vlen
                        st["min"] = vlen if st["min"] is None else min(st["min"], vlen)
                        st["max"] = vlen if st["max"] is None else max(st["max"], vlen)
            else:
                # If rows aren't dicts, track the overall row "type" under a special key
                type_counts["__row__"][safe_type(obj)] += 1

    # Report
    print("=" * 80)
    print("JSONL INSPECTOR")
    print("=" * 80)
    print(f"Path: {path}")
    print(f"File size: {human_bytes(file_size)} ({file_size:,} bytes)")
    print(f"Rows scanned: {total_rows:,}" + (f" (limited to {MAX_ROWS_SCAN:,})" if MAX_ROWS_SCAN else ""))
    print(f"Empty/blank lines skipped: {empty_lines:,}")
    print(f"JSON parse failures: {bad_json_rows:,}")
    print()

    # Preview
    print("-" * 80)
    print(f"PREVIEW: first {N_PREVIEW} rows")
    print("-" * 80)
    if PRINT_PREVIEW_AS_JSON and preview_parsed:
        for idx, obj in enumerate(preview_parsed, 1):
            print(f"\nRow {idx}:")
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:5000])  # safety cap
    else:
        for idx, line in enumerate(preview_raw, 1):
            print(f"{idx}: {line[:2000]}")  # safety cap
    print()

    # Schema summary
    if keys_seen:
        print("-" * 80)
        print("SCHEMA SUMMARY")
        print("-" * 80)
        print(f"Unique keys found: {len(keys_seen):,}")
        print()

        # show most common keys
        print("Most common keys (key: presence% / count):")
        for k, c in key_counts.most_common(30):
            pct = (c / total_rows * 100) if total_rows else 0
            print(f"  {k}: {pct:6.2f}%  ({c:,})")
        print()

        # Per-key type distribution (top 30 keys)
        print("Type distribution for top keys:")
        for k, _ in key_counts.most_common(20):
            tc = type_counts[k]
            type_str = ", ".join([f"{t}:{n}" for t, n in tc.most_common(6)])
            print(f"  {k}: {type_str}")
        print()

        # Length stats for str/list/dict fields
        print("Length stats (only for str/list/dict fields):")
        shown = 0
        for k, st in length_stats.items():
            if st["count"] == 0:
                continue
            avg = st["sum"] / st["count"]
            print(f"  {k}: count={st['count']:,}  min={st['min']}  avg={avg:.2f}  max={st['max']}")
            shown += 1
            if shown >= 30:
                print("  ... (truncated)")
                break
        print()
    else:
        # Non-dict rows
        if "__row__" in type_counts:
            print("-" * 80)
            print("ROW TYPE SUMMARY (rows are not JSON objects/dicts)")
            print("-" * 80)
            tc = type_counts["__row__"]
            for t, n in tc.most_common():
                pct = (n / total_rows * 100) if total_rows else 0
                print(f"  {t}: {pct:6.2f}% ({n:,})")
            print()

    print("=" * 80)
    print("Done.")

if __name__ == "__main__":
    jsonl_inspect(file_path)