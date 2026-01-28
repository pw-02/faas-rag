import argparse
import time
import csv
import os
import numpy as np
import faiss
from tqdm import tqdm
import json
import sqlite3
from typing import List, Optional, Dict

from experiments.memory_monitor import MemoryMonitor, bytes_to_gb
from experiments.cpu_monitor import CPUMonitor


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Query Benchmark")
    parser.add_argument(
        "--index-name",
        type=str,
        default="data/indexes/index_cc_monolithic_fiass_100k/index_cc_monolithic_100k.faiss",
        help="Path to the FAISS index file",
    )
    parser.add_argument("--nprobe", type=int, nargs="+", default=[256], help="List of nprobe values")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[32], help="List of batch sizes")
    parser.add_argument("--max-batches", type=int, default=50, help="Maximum number of batches to process")

    parser.add_argument(
        "--memory-monitor-interval",
        type=float,
        default=0.05,
        help="Interval (s) between memory samples",
    )
    parser.add_argument(
        "--cpu-monitor-interval",
        type=float,
        default=0.05,
        help="Interval (s) between CPU samples",
    )

    parser.add_argument(
        "--queries",
        type=str,
        default="data/triviaqa/triviaqa_encodings.npy",
        help="Path to .npy embeddings",
    )
    parser.add_argument("--retrieved-docs", type=int, nargs="+", default=[5], help="List of top-k to retrieve")
    parser.add_argument("--num-threads", type=int, nargs="+", default=[1], help="List of FAISS OMP thread counts")
    parser.add_argument("--output-dir", type=str, default="data/profiling/", help="Directory for results CSV")

    parser.add_argument(
        "--doc-store",
        nargs="+",
        choices=["none", "sqlite", "memory_jsonl"],
        default=["none", "sqlite", "memory_jsonl"],
        help="Doc fetch backends to benchmark",
    )
    parser.add_argument(
        "--docs-db",
        type=str,
        default="data/indexes/index_cc_monolithic_fiass_100k/cc_docs_100k.db",
        help="SQLite docs DB path",
    )
    parser.add_argument(
        "--docs-jsonl",
        type=str,
        default="data/indexes/index_cc_monolithic_fiass_100k/cc_docs_100k.jsonl",
        help="Docs JSONL path",
    )
    parser.add_argument("--fetch-topk", type=int, default=None, help="Fetch only top-k of retrieved docs")
    return parser.parse_args()


class DocStore:
    def get_many(self, row_ids: List[int]) -> List[str]:
        raise NotImplementedError


class NoneDocStore(DocStore):
    def get_many(self, row_ids: List[int]) -> List[str]:
        return []


class SqliteByRowDocStore(DocStore):
    # expects docs(row_id INTEGER PRIMARY KEY, text TEXT)
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA cache_size = 100;")
        self.conn.execute("PRAGMA mmap_size = 0;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")

    def get_many(self, row_ids: List[int]) -> List[str]:
        if not row_ids:
            return []
        q = ",".join("?" * len(row_ids))
        cur = self.conn.cursor()
        cur.execute(f"SELECT row_id, text FROM docs WHERE row_id IN ({q})", row_ids)
        rows = cur.fetchall()
        m = {int(rid): text for rid, text in rows}
        return [m[rid] for rid in row_ids if rid in m]


class MemoryJsonlByRowDocStore(DocStore):
    # expects {"row_id": int, "text": str} per line
    def __init__(self, jsonl_path: str):
        temp: Dict[int, str] = {}
        max_row = -1
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                rid = int(o["row_id"])
                temp[rid] = o["text"]
                max_row = max(max_row, rid)

        self.texts: List[Optional[str]] = [None] * (max_row + 1)
        for rid, txt in temp.items():
            self.texts[rid] = txt

    def get_many(self, row_ids: List[int]) -> List[str]:
        out = []
        for rid in row_ids:
            if 0 <= rid < len(self.texts):
                t = self.texts[rid]
                if t is not None:
                    out.append(t)
        return out


def build_doc_store(kind: str, docs_db: str = None, docs_jsonl: str = None) -> DocStore:
    if kind == "none":
        return NoneDocStore()
    if kind == "sqlite":
        if not docs_db:
            raise ValueError("--docs-db is required when --doc-store sqlite")
        return SqliteByRowDocStore(docs_db)
    if kind == "memory_jsonl":
        if not docs_jsonl:
            raise ValueError("--docs-jsonl is required when --doc-store memory_jsonl")
        return MemoryJsonlByRowDocStore(docs_jsonl)
    raise ValueError(f"Unknown doc store: {kind}")


def load_faiss_index(index_name: str, nprobe: int):
    index = faiss.read_index(index_name)
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    return index


def perform_queries(
    index,
    doc_store,
    retrieved_docs: int,
    embeddings: np.ndarray,
    batch_size: int,
    fetch_topk: Optional[int] = None,
    max_batches: int = 1000,
    memory_monitor_interval: float = 0.05,
    cpu_monitor_interval: float = 0.05,
):
    search_times = []
    fetch_times = []
    total_times = []

    if fetch_topk is None:
        fetch_topk = retrieved_docs

    # number of batches we will actually run
    total_batches = min(len(embeddings) // batch_size, max_batches)

    mem = MemoryMonitor(interval_s=memory_monitor_interval).start()
    cpu = CPUMonitor(interval_s=cpu_monitor_interval).start()
    cpu_time_start = time.process_time()

    try:
        for idx in tqdm(
            range(0, min(len(embeddings), max_batches * batch_size), batch_size),
            desc="Querying batches",
            leave=False,
            position=4,
        ):
            batch = embeddings[idx : idx + batch_size]

            t0 = time.perf_counter()
            _, I = index.search(batch, retrieved_docs)
            t1 = time.perf_counter()

            tf0 = time.perf_counter()
            if doc_store is not None:
                for j in range(I.shape[0]):
                    row_ids = [int(x) for x in I[j, :fetch_topk].tolist() if int(x) >= 0]
                    _docs = doc_store.get_many(row_ids)
            tf1 = time.perf_counter()

            search_times.append(t1 - t0)
            fetch_times.append(tf1 - tf0)
            total_times.append(tf1 - t0)

        cpu_seconds = time.process_time() - cpu_time_start

    finally:
        cpu.stop()
        mem.stop()

    def avg(xs): return (sum(xs) / len(xs)) if xs else 0.0

    return (
        avg(search_times),
        avg(fetch_times),
        avg(total_times),
        total_batches,
        mem.avg_rss_bytes,
        mem.peak_rss,
        cpu.avg_cpu_percent,
        cpu.peak_cpu_percent,
        cpu_seconds,
    )


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "retrieval_monolithic_latency.csv")

    index = load_faiss_index(args.index_name, args.nprobe[0])
    embeddings = np.load(args.queries)

    print(f"Total queries loaded: {embeddings.shape[0]}")

    fieldnames = [
        "Index Name", "Doc Store", "nprobe", "Batch Size", "Batches Processed",
        "Retrieved Docs", "Fetch TopK", "Num Threads",
        "Avg Search Time (s)", "Avg Fetch Time (s)", "Avg Total Time (s)",
        "Search ms/query", "Fetch ms/query", "Total ms/query",
        "Avg RSS (GB)", "Peak RSS (GB)",
        "Avg CPU Percent", "Peak CPU Percent",
        "CPU Time (s)", "CPU ms/query",
    ]

    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for doc_store_kind in tqdm(args.doc_store, desc="Doc store", position=0):
            doc_store = build_doc_store(doc_store_kind, docs_db=args.docs_db, docs_jsonl=args.docs_jsonl)

            for nprobe in tqdm(args.nprobe, desc="nprobe values", position=1, leave=False):
                if hasattr(index, "nprobe"):
                    index.nprobe = nprobe

                for batch_size in tqdm(args.batch_size, desc=f"Batch sizes (nprobe={nprobe})", position=2, leave=False):
                    for retrieved_docs in tqdm(
                        args.retrieved_docs,
                        desc=f"Retrieved Docs (nprobe={nprobe}, batch_size={batch_size})",
                        position=3,
                        leave=False,
                    ):
                        fetch_topk = args.fetch_topk or retrieved_docs

                        for num_threads in tqdm(
                            args.num_threads,
                            desc=f"Num Threads (nprobe={nprobe}, batch_size={batch_size}, retrieved_docs={retrieved_docs})",
                            position=4,
                            leave=False,
                        ):
                            faiss.omp_set_num_threads(num_threads)

                            (
                                avg_search_s, avg_fetch_s, avg_total_s, total_batches,
                                avg_rss_bytes, peak_rss,
                                avg_cpu_percent, peak_cpu_percent,
                                cpu_seconds
                            ) = perform_queries(
                                index=index,
                                doc_store=doc_store,
                                retrieved_docs=retrieved_docs,
                                embeddings=embeddings,
                                batch_size=batch_size,
                                fetch_topk=fetch_topk,
                                max_batches=args.max_batches,
                                memory_monitor_interval=args.memory_monitor_interval,
                                cpu_monitor_interval=args.cpu_monitor_interval,
                            )

                            search_ms_per_query = (avg_search_s * 1000.0) / batch_size
                            fetch_ms_per_query = (avg_fetch_s * 1000.0) / batch_size
                            total_ms_per_query = (avg_total_s * 1000.0) / batch_size

                            total_queries = total_batches * batch_size
                            cpu_ms_per_query = (cpu_seconds * 1000.0) / total_queries if total_queries else 0.0

                            writer.writerow({
                                "Index Name": args.index_name,
                                "Doc Store": doc_store_kind,
                                "nprobe": nprobe,
                                "Batch Size": batch_size,
                                "Batches Processed": total_batches,
                                "Retrieved Docs": retrieved_docs,
                                "Fetch TopK": fetch_topk,
                                "Num Threads": num_threads,
                                "Avg Search Time (s)": avg_search_s,
                                "Avg Fetch Time (s)": avg_fetch_s,
                                "Avg Total Time (s)": avg_total_s,
                                "Search ms/query": search_ms_per_query,
                                "Fetch ms/query": fetch_ms_per_query,
                                "Total ms/query": total_ms_per_query,
                                "Avg RSS (GB)": bytes_to_gb(avg_rss_bytes),
                                "Peak RSS (GB)": bytes_to_gb(peak_rss),
                                "Avg CPU Percent": avg_cpu_percent,
                                "Peak CPU Percent": peak_cpu_percent,
                                "CPU Time (s)": cpu_seconds,
                                "CPU ms/query": cpu_ms_per_query,
                            })
                            file.flush()

    print(f"\nWrote results to: {output_file}")


if __name__ == "__main__":
    main()
