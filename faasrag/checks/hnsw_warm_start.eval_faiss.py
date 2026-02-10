"""
If two queries are close in embedding space, can I reuse part of the HNSW search from 
the first query to speed up the second?

For every query:
- HNSW starts from the same global entry node.
- It explores the graph using a beam (efSearch).
- It eventually finds nearest neighbors.
- Each query is totally independent.

Then:
- Instead of always starting from the same entry node:
- You sometimes start from nodes that worked well for the previous query.
"""

