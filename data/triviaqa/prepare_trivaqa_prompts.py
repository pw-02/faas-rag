from datasets import load_dataset
import csv

def main():
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    split = "train"  # or "validation"
    output_path = f"triviaqa_{split}_qa.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "question",
            "answer_value",
            "answer_aliases",
        ])

        for ex in ds[split]:
            writer.writerow([
                ex["question_id"],
                ex["question"],
                ex["answer"]["value"],
                "|".join(ex["answer"]["aliases"]),
            ])

    print(f"Wrote {len(ds[split])} QA pairs to {output_path}")


if __name__ == "__main__":
    main()
