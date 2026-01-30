#!/usr/bin/env python3
import argparse
import numpy as np
from datasets import load_dataset
import boto3


# Mapping of allowed index sizes to dataset names.
DATASET_MAPPING = {
    "100k": "mohdumar/SPHERE_100K",
    "100m": "mohdumar/SPHERE_100M",
    "899m": "mohdumar/SPHERE_899M",
}

s3vectors = boto3.client("s3vectors", region_name="us-west-2")


def flush_batch(vectorBucketName: str, indexName: str, batch: list[dict]) -> None:
    """Send one batch to S3 Vectors."""
    if not batch:
        return
    s3vectors.put_vectors(
        vectorBucketName=vectorBucketName,
        indexName=indexName,
        vectors=batch,
    )


def populate_s3_vector_index_from_dataset(
    dataset_name: str,
    vectorBucketName: str,
    s3_index_name: str,
    dim: int = 768,
    dataset_streaming: bool = False,
    batch_size: int = 500,
) -> None:
    print(f"Loading Hugging Face dataset: {dataset_name} (streaming={dataset_streaming}) ...")
    ds = load_dataset(dataset_name, split="train", streaming=dataset_streaming)

    batch: list[dict] = []
    count = 0

    for ex in ds:
        if "vector" not in ex:
            raise ValueError("Example missing 'vector' field in dataset.")
        # Prefer a stable ID from the dataset
        if "id" not in ex:
            raise ValueError("Example missing 'id' field in dataset.")
        
        vec = np.asarray(ex["vector"], dtype=np.float32)

        # Expect shape (dim,)
        if vec.ndim != 1 or vec.shape[0] != dim:
            raise ValueError(
                f"Vector dim mismatch at row {count}: expected ({dim},), got {tuple(vec.shape)}"
            )
        key = str(ex["id"])
        # S3 Vectors expects: key + data.float32 (NOT raw list of floats)
        batch.append({
            "key": key,  # If dataset has a stable id, use it instead
            "data": {"float32": vec.tolist()},
            # "metadata": {"source": dataset_name},  # optional
        })

        count += 1

        if len(batch) >= batch_size:
            start = count - len(batch)
            end = count - 1
            print(f"Putting vectors [{start}..{end}] (batch size={len(batch)})")
            flush_batch(vectorBucketName, s3_index_name, batch)
            batch = []

    # Final partial batch
    if batch:
        start = count - len(batch)
        end = count - 1
        print(f"Putting final batch [{start}..{end}] (batch size={len(batch)})")
        flush_batch(vectorBucketName, s3_index_name, batch)

    print(f"Done. Total vectors written: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Populate an S3 Vectors index using a Hugging Face dataset"
    )
    parser.add_argument(
        "--index-size",
        type=str,
        required=False,
        default="100k",
        help="Index size. Allowed values: 100k, 100m, 899m",
    )
    parser.add_argument(
        "--dataset-streaming",
        action="store_true",
        help="Enable dataset streaming (recommended for large datasets)",
    )
    parser.add_argument(
        "--vector-bucket-name",
        type=str,
        required=False,
        default="rag-vector-indexes",
        help="S3 Vector Bucket Name where vectors will be stored",
    )
    parser.add_argument(
        "--s3-index-name",
        type=str,
        required=False,
        default="index-cc-monolithic-100k",
        help="S3 Vector Index Name to create/populate",
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=False,
        default=768,
        help="Dimension of the vectors (default: 768)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        default=500 ,
        help=f"Vectors per put_vectors call (default: {500 })",
    )

    args = parser.parse_args()

    index_size = args.index_size.lower()
    if index_size not in DATASET_MAPPING:
        raise ValueError("Invalid index size. Choose one of: 100k, 100m, 899m")

    dataset_name = DATASET_MAPPING[index_size]

    populate_s3_vector_index_from_dataset(
        dataset_name=dataset_name,
        vectorBucketName=args.vector_bucket_name,
        s3_index_name=args.s3_index_name,
        dim=args.dim,
        dataset_streaming=args.dataset_streaming,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
