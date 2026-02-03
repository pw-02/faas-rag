import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import faiss  # pip install faiss-cpu (or faiss-gpu)

def parse_args():
    p = argparse.ArgumentParser(description="Build a FAISS index from facebook/wiki_dpr precomputed embeddings.")
    p.add_argument("--dataset", default="facebook/wiki_dpr")
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true", default=False)
    p.add_argument("--no-streaming", dest="streaming", action="store_false")

    p.add_argument("--dataset_name", default="psgs_w100.nq.no_index",
                   help="e.g. psgs_w100.nq.no_index or psgs_w100.multiset.no_index")

    p.add_argument("--index_type", choices=["flat_ip", "ivf_ip", "hnsw_ip"], default="flat_ip")
    p.add_argument("--n_vectors", type=int, default=100_000)
    p.add_argument("--batch_size", type=int, default=8192)

    # IVF
    p.add_argument("--n_lists", type=int, default=4096)
    p.add_argument("--train_size", type=int, default=50_000)
    p.add_argument("--nprobe", type=int, default=16, help="IVF: number of lists to probe at search time")

    # HNSW
    p.add_argument("--n_neighbors", type=int, default=32)
    p.add_argument("--ef_construction", type=int, default=200)
    p.add_argument("--ef_search", type=int, default=64)

    p.add_argument("--out_dir", default="faiss_wiki_dpr")
    p.add_argument("--store_text", action="store_true", default=False,
                   help="Store full passage text in meta.jsonl")
    p.add_argument("--snippet_chars", type=int, default=0,
                   help="Store a snippet of text of this length (0 disables)")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    safe_cfg = args.dataset_name.replace(".", "_")
    index_path = os.path.join(args.out_dir, f"index_{safe_cfg}_{args.index_type}_{args.n_vectors}.faiss")
    meta_path  = os.path.join(args.out_dir, f"meta_{safe_cfg}_{args.n_vectors}.jsonl")

    print("=== Config ===")
    print("dataset      :", args.dataset)
    print("dataset_name :", args.dataset_name)
    print("split        :", args.split)
    print("streaming    :", args.streaming)
    print("index_type   :", args.index_type)
    print("n_vectors    :", args.n_vectors)
    print("batch_size   :", args.batch_size)
    print("out_dir      :", args.out_dir)
    print("store_text   :", args.store_text)
    print("snippet_chars:", args.snippet_chars)
    if args.index_type == "ivf_ip":
        print("n_lists      :", args.n_lists)
        print("train_size   :", args.train_size)
        print("nprobe       :", args.nprobe)
    if args.index_type == "hnsw_ip":
        print("n_neighbors  :", args.n_neighbors)
        print("efConstruct  :", args.ef_construction)
        print("efSearch     :", args.ef_search)
    print("==============\n")

    # Load streaming dataset
    ds = load_dataset(args.dataset, name=args.dataset_name, split=args.split, streaming=args.streaming)
    
    first = next(iter(ds))
    if "embeddings" not in first:
        raise KeyError(f"Expected 'embeddings' field but got keys: {list(first.keys())}")
    dim = len(first["embeddings"])
    
    print("Embedding dim:", dim)

    # Re-create iterator
    ds = load_dataset(args.dataset, name=args.dataset_name, split=args.split, streaming=args.streaming)

    # Build FAISS index
    if args.index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)

    elif args.index_type == "ivf_ip":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, args.n_lists, faiss.METRIC_INNER_PRODUCT)

    elif args.index_type == "hnsw_ip":
        index = faiss.IndexHNSWFlat(dim, args.n_neighbors, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.ef_construction
        index.hnsw.efSearch = args.ef_search

    else:
        raise ValueError("Unexpected index_type")

    if args.index_type == "ivf_ip":
        index.nprobe = args.nprobe

    buf_vecs, buf_meta = [], []
    added = 0

    train_vecs = []
    need_train = (args.index_type == "ivf_ip")

    with open(meta_path, "w", encoding="utf-8") as mf:
        for row in tqdm(ds, total=args.n_vectors, desc=f"Indexing {args.dataset_name}"):
            if added >= args.n_vectors:
                break

            v = np.asarray(row["embeddings"], dtype=np.float32)
            if v.shape[0] != dim:
                raise ValueError(f"Dim mismatch: got {v.shape[0]} expected {dim}")

            # DPR NOTE: do NOT normalize

            if need_train and len(train_vecs) < args.train_size:
                train_vecs.append(v)

            faiss_id = added + len(buf_vecs)
            buf_vecs.append(v)

            meta = {
                "faiss_id": faiss_id,
                "id": row.get("id"),
                "title": row.get("title", ""),
            }
            if args.snippet_chars > 0:
                meta["snippet"] = (row.get("text", "") or "")[: args.snippet_chars]
            if args.store_text:
                meta["text"] = row.get("text", "") or ""
            buf_meta.append(meta)

            if len(buf_vecs) >= args.batch_size:
                X = np.vstack(buf_vecs).astype(np.float32)

                if need_train and not index.is_trained:
                    T = np.vstack(train_vecs).astype(np.float32)
                    print(f"\nTraining IVF with {T.shape[0]} vectors...")
                    index.train(T)

                index.add(X)

                for m in buf_meta:
                    mf.write(json.dumps(m, ensure_ascii=False) + "\n")

                added += len(buf_vecs)
                buf_vecs.clear()
                buf_meta.clear()

        # Flush remainder
        if buf_vecs:
            X = np.vstack(buf_vecs).astype(np.float32)

            if need_train and not index.is_trained:
                T = np.vstack(train_vecs).astype(np.float32)
                print(f"\nTraining IVF with {T.shape[0]} vectors...")
                index.train(T)

            index.add(X)

            for m in buf_meta:
                mf.write(json.dumps(m, ensure_ascii=False) + "\n")

            added += len(buf_vecs)

    faiss.write_index(index, index_path)

    print("\nDone.")
    print("Vectors indexed:", index.ntotal)
    print("Index saved to :", index_path)
    print("Meta saved to  :", meta_path)

if __name__ == "__main__":
    main()
