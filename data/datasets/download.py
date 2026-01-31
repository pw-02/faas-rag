from datasets import load_dataset
import os


qa_datasets = ["nq","triviaqa","wikiqa","squad"]
multi_hop_datasets = ["hotpotqa","musique","2wikimultihopqa"]
long_form_datasets = ["eli5","asqa"]
multiple_choice_datasets = ["arc","hellaswag","openbookqa"]
fact_verification_datasets = ["fever"]

category_datasets = {
    "qa": qa_datasets,
    "multi_hop": multi_hop_datasets,
    "long_form": long_form_datasets,
    "multiple_choice": multiple_choice_datasets,
    "fact_verification": fact_verification_datasets
}

for category, datasets in category_datasets.items():
    for name in datasets:
        output_train = f"data/datasets/{category}/{name}/{name}_train.jsonl"
        output_dev = f"data/datasets/{category}/{name}/{name}_dev.jsonl"
        output_test = f"data/datasets/{category}/{name}/{name}_test.jsonl"

        if os.path.exists(output_train) or os.path.exists(output_dev) or os.path.exists(output_test):
            print(f"{name} dataset already exists. Skipping download.")
            continue
        
        ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", name)
        os.makedirs(os.path.dirname(output_train), exist_ok=True)
        os.makedirs(os.path.dirname(output_dev), exist_ok=True)
        os.makedirs(os.path.dirname(output_test), exist_ok=True)
        ds["train"].to_json(output_train) if "train" in ds else None
        ds["dev"].to_json(output_dev) if "dev" in ds else None
        ds["test"].to_json(output_test) if "test" in ds else None
        print(f"Downloaded {name} dataset to data/datasets/{category}/{name}/")
