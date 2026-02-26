import faiss
import numpy as np
import os


def describe_faiss_index(index, name="FAISS Index"):
    print(f"\n===== {name} =====")

    # Basic info
    print("Type:", type(index))
    print("Dimension (d):", index.d)
    print("Total vectors (ntotal):", index.ntotal)
    print("Is trained:", index.is_trained)

    # Metric type
    metric_map = {
        faiss.METRIC_L2: "L2 (Euclidean)",
        faiss.METRIC_INNER_PRODUCT: "Inner Product"
    }
    print("Metric:", metric_map.get(index.metric_type, index.metric_type))

    # Try IVF info
    if isinstance(index, faiss.IndexIVF):
        print("\n--- IVF Info ---")
        print("nlist (clusters):", index.nlist)
        print("nprobe (search probes):", index.nprobe)

        invlists = index.invlists
        if invlists:
            total_codes = sum(invlists.list_size(i) for i in range(index.nlist))
            print("Total inverted list entries:", total_codes)

    # Try PQ info
    if isinstance(index, faiss.IndexIVFPQ) or isinstance(index, faiss.IndexPQ):
        print("\n--- PQ Info ---")
        pq = index.pq
        print("PQ M (subquantizers):", pq.M)
        print("PQ bits per code:", pq.nbits)
        print("Code size (bytes):", index.code_size)

    # Try HNSW info
    if isinstance(index, faiss.IndexHNSW):
        print("\n--- HNSW Info ---")
        print("HNSW M:", index.hnsw.M)
        print("efConstruction:", index.hnsw.efConstruction)
        print("efSearch:", index.hnsw.efSearch)

    print("\n===== End =====\n")


def estimate_index_size(index):
    """Very rough RAM usage estimate."""
    if hasattr(index, "ntotal"):
        if hasattr(index, "code_size"):
            return index.ntotal * index.code_size
        else:
            return index.ntotal * index.d * 4
    return None

if __name__ == "__main__":
    # Example usage
    index_path = "corpus/faiss_wiki_dpr/ivf_all/index_psgs_w100_nq_no_index_ivf_ip_all.faiss"
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        describe_faiss_index(index, name=os.path.basename(index_path))
        size_estimate = estimate_index_size(index)
        if size_estimate:
            print(f"Estimated RAM usage: {size_estimate / (1024**2):.2f} MB")
    else:
        print(f"Index file not found: {index_path}")