import os
import numpy as np
import diskannpy as dap

# demo data
X = np.random.rand(10_000, 128).astype(np.float32)
Q = np.random.rand(10, 128).astype(np.float32)

os.makedirs("myindex", exist_ok=True)

# Build index (in-memory)
dap.build_memory_index(
    data=X,
    distance_metric="l2",     # "l2" or "cosine"
    index_directory="myindex",
    index_prefix="ann",
    complexity=100,           # build quality/speed tradeoff
    graph_degree=64,          # memory/recall tradeoff
    num_threads=0,            # 0 = use all cores
)

# Load and search
index = dap.StaticMemoryIndex(
    index_directory="myindex",
    index_prefix="ann",
    num_threads=0,                 # 0 = use all cores
    initial_search_complexity=50   # starting L / complexity
)

ids, dists = index.batch_search(Q, k_neighbors =10, complexity=50, num_threads=0)

print("Top-10 IDs for first query:", ids[0])
print("Top-10 distances for first query:", dists[0])
print("✅ DiskANN works")
