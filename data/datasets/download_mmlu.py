import os, json
from datasets import load_dataset, get_dataset_config_names

# dataset = load_dataset("cais/mmlu", "all")  # config = "all"
out_dir = "data/datasets/mmlu"
os.makedirs(out_dir, exist_ok=True)
subjects = get_dataset_config_names("cais/mmlu")

# Often includes "all" as well — we usually skip it here:
subjects = [s for s in subjects]

for subject in subjects:
    subject_ds = load_dataset("cais/mmlu", subject)  # DatasetDict
    subject_dir = os.path.join(out_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)

    for split, ds in subject_ds.items():
        path = os.path.join(subject_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote subject: {subject}")  

print("Done!")