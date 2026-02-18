import json

# ---- CONFIG ----
input_jsonl = "data/datasets/qa/nq/nq_train.jsonl"
output_jsonl = "data/datasets/nq_train_filtered.jsonl"
ids_file = "data/datasets/target_ids.txt"  # optional: load ids from a file (one id per line)
# your selected ids (one per line or copy/paste)

#load ids into list from a file
with open(ids_file, "r", encoding="utf-8") as f:
    selected_ids = [line.strip() for line in f if line.strip()]

selected_ids = set(selected_ids)  # faster lookup

kept = 0

with open(input_jsonl, "r", encoding="utf-8") as fin, open(
    output_jsonl, "w", encoding="utf-8"
) as fout:
    for line in fin:
        if not line.strip():
            continue

        obj = json.loads(line)

        if obj.get("id") in selected_ids:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

print(f"Saved {kept} rows → {output_jsonl}")
