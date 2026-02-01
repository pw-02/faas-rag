from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
# from core.rag_pipeline_single_node import RagPipelineSingleNode
from core.rag_pipeline import RagPipelineBase
from core.proximity_cache import ProximityCache
from core.rag_profile_utils import (
    load_questions_from_jsonl,
    save_batch_results_csv,
    create_summary_from_csvs,
)

DEBUG_MODE = True

def debug_args():
    """
    Debug-only arguments for profiling.
    This file should NEVER be used for real experiments.
    """
    return SimpleNamespace(
        # choose pipeline
        cache = "proximity",  # "none" or "proximity"

        # inputs
        queries_file="data/datasets/qa/triviaqa/triviaqa_dev.jsonl",
        queries_column="question",
        docstore_path="data/indexes/sphere/cc_docs_100k.jsonl",
        index=["data/indexes/synthetic/flat_ip_d768_n100000_norm1.index"],
        out_dir="results/debug",

        # runtime limits (KEEP SMALL)
        batch_size=1,
        max_batches=1000,
        show_progress=True,

        # retrieval / generation
        top_k=5,
        max_context_docs=3,
        max_new_tokens=128,
        
        generator="simulated", #Options: simulated, tiny-gpt2, gpt2
        sim_generation_delay_s=0.01,
        embedder="BAAI/bge-base-en-v1.5", #Optionns: synthetic, BAAI/bge-base-en-v1.5
        embedder_max_length=512,
        do_sample=True,

        # perf / faiss
        num_faiss_threads=4,
        n_probe=None,

        # proximity cache (ignored if pipeline="single")
        cache_policy="fifo",
        tolerance=0.2,
        cache_size=1000,
        lsh_cache_num_hash=64,
        lsh_cache_expected_dim=0,  # 0 = infer from index.d
        lsh_cache_bucket_capacity=10,
        seed=42,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile RAG pipeline over one or more FAISS indexes.")

    # inputs
    p.add_argument("--queries-file", required=True, help="JSONL file containing questions.")
    p.add_argument("--queries-column", default="question", help="Column name in queries JSONL.")
    p.add_argument("--docstore-path", required=True, help="Path to docstore (jsonl, etc.).")
    p.add_argument("--index", action="append", required=True,
                   help="FAISS index path. Repeat --index for multiple.")
    p.add_argument("--out-dir", default="results/rag_profile")

    # choose pipeline
    p.add_argument("--cache", choices=["None", "proximity"], default="None",
                   help="Which pipeline implementation to profile.")

    # workload sizing
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=10)

    # retrieval/generation config
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-context-docs", type=int, default=3)
    p.add_argument("--generator", default="synthetic")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--do-sample", action="store_true", default=False)
    p.add_argument("--embedder", default="synthetic")
    p.add_argument("--embedder-max-length", type=int, default=512)

    # perf / faiss
    p.add_argument("--sim-generation-delay-s", type=float, default=0.05)
    p.add_argument("--n-probe", type=int, default=None,
                   help="FAISS IVF nprobe. If omitted, script will set 256 when index path contains 'ivf'.")
    p.add_argument("--show-progress", action="store_true", default=False)
    p.add_argument("--num-faiss-threads", type=int, default=None, help="If set, configures FAISS to use this many threads.")

    # proximity cache knobs (only used if --pipeline proximity)
    p.add_argument("--cache-policy", default=None, help="None|fifo|lru|lsh_fifo|lsh_lru")
    p.add_argument("--cache-size", type=int, default=100)
    p.add_argument("--lsh-cache-num-hash", type=int, default=128)
    p.add_argument("--tolerance", type=float, default=0.7)
    p.add_argument("--lsh-cache-expected-dim", type=int, default=0,
                   help="0 means infer from FAISS index dimension.")
    p.add_argument("--lsh-cache-bucket-capacity", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:

    if DEBUG_MODE:
        args = debug_args()
    else:
        args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_questions_from_jsonl(
        args.queries_file,
        column=args.queries_column,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    num_runs = 2

    for _ in range(num_runs):
        csv_results: list[str] = []

        for index_path in args.index:
            index_p = Path(index_path)
            index_id = index_p.stem
            csv_path = str(out_dir / f"{args.cache}__{index_id}.csv")

            # auto nprobe only for IVF (optional)
            n_probe = args.n_probe
            if n_probe is None and "ivf" in index_path.lower():
                n_probe = 256

            print(f"\n--- Running {args.cache} pipeline with index: {index_path} ---")

            if args.cache == "proximity":
                cache = ProximityCache(
                    cache_policy=args.cache_policy,
                    tolerance=args.tolerance,
                    cache_size=args.cache_size,
                    lsh_num_hash=args.lsh_cache_num_hash,
                    lsh_bucket_capacity=args.lsh_cache_bucket_capacity,
                    seed=args.seed,
                )
            else:
                cache = None

            pipeline = RagPipelineBase(
                generator_name=args.generator,
                embedder_name=args.embedder,
                vector_index_path=index_path,
                docstore_path=args.docstore_path,
                device=None,
                top_k=args.top_k,
                show_progress=args.show_progress,
                n_probe=n_probe,
                num_faiss_threads=args.num_faiss_threads,
                batch_size=args.batch_size,
                max_context_docs=args.max_context_docs,
                max_new_tokens=args.max_new_tokens,
                embedder_max_length=args.embedder_max_length,
                do_sample=args.do_sample,
                simulated_generation_delay_s=args.sim_generation_delay_s,
                cache=cache,
            )

            batch_results = pipeline.run(queries, return_prompt=False, return_contexts=False)

            # run() returns dict for a single query; ensure list
            if isinstance(batch_results, dict):
                batch_results = [batch_results]

            if cache is not None:
                print(f"Cache stats: hits={cache.cache_hit_count}, misses={cache.cache_miss_count}")
                cache_stats = cache.get_stats()
            else:
                cache_stats = None

            save_batch_results_csv(batch_results, cache_stats, csv_path)
            csv_results.append(csv_path)

        summary_df = create_summary_from_csvs(csv_results, str(out_dir / f"summary__{args.cache}.csv"))
        # print("\n=== Summary ===")
        # print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
