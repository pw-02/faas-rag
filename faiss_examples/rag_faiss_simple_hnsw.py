import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    doc_id: int


def chunk_text(text: str, chunk_size: int = 240, overlap: int = 60) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_faiss_index_hnsw(
    embed_model: SentenceTransformer,
    docs: List[str],
    chunk_size: int = 240,
    overlap: int = 60,
    M: int = 32,
    ef_construction: int = 80,
) -> Tuple[faiss.Index, List[Chunk], int]:
    """
    Returns (faiss_index, chunks, embedding_dim)

    HNSW notes:
      - No training needed.
      - M controls graph connectivity (memory/quality).
      - efConstruction controls build quality/time.
      - efSearch controls query quality/time.
      - Using IP + normalized embeddings => cosine similarity.
    """
    chunks: List[Chunk] = []
    for doc_id, doc_text in enumerate(docs):
        for c in chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(text=c, doc_id=doc_id))

    texts = [c.text for c in chunks]
    xb = embed_model.encode(texts, normalize_embeddings=True, batch_size=64)
    xb = np.asarray(xb, dtype="float32")
    dim = xb.shape[1]

    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction

    index.add(xb)
    return index, chunks, dim


def retrieve(
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    query: str,
    top_k: int = 3,
    ef_search: int = 64,
) -> List[Dict[str, Any]]:
    # HNSW search-time knob
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search

    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        ch = chunks[int(idx)]
        results.append({"score": float(score), "chunk": ch.text, "doc_id": ch.doc_id})
    return results


def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        [f"[{i}] (doc {r['doc_id']}, score {r['score']:.3f}) {r['chunk']}"
         for i, r in enumerate(retrieved)]
    )
    return (
        "You are a helpful assistant. Use ONLY the context below to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (grounded in context):"
    )


def answer_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text


def fallback_answer(retrieved: List[Dict[str, Any]]) -> str:
    lines = ["(No API key set, so this is a retrieval-only fallback.)", ""]
    lines.append("Most relevant context:")
    for i, r in enumerate(retrieved):
        lines.append(f"- [{i}] {r['chunk']}")
    return "\n".join(lines)


def main() -> None:
    docs = [
        "FAISS supports HNSW (Hierarchical Navigable Small World) indexes for approximate nearest neighbor search.",
        "HNSW builds a graph and does not require training like IVF.",
        "HNSW parameters include M (connectivity) and efSearch (query-time accuracy vs speed).",
        "Normalize embeddings and use inner product for cosine similarity search.",
        "Chunking helps retrieval by splitting long docs into smaller semantically meaningful passages.",
    ]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index, chunks, dim = build_faiss_index_hnsw(
        embed_model,
        docs,
        chunk_size=240,
        overlap=60,
        M=32,
        ef_construction=80,
    )
    print(f"Built FAISS HNSW index with {index.ntotal} vectors (dim={dim}).")

    query = "Does HNSW require training and what parameters matter?"
    retrieved = retrieve(embed_model, index, chunks, query, top_k=3, ef_search=64)

    print("\n=== Retrieved ===")
    for r in retrieved:
        print(f"- score={r['score']:.3f} doc={r['doc_id']} chunk={r['chunk']}")

    prompt = build_prompt(query, retrieved)

    if os.getenv("OPENAI_API_KEY"):
        print("\n=== Prompt to LLM ===")
        print(prompt)
        print("\n=== LLM Answer ===")
        print(answer_with_openai(prompt))
    else:
        print("\n=== Retrieval-only Answer ===")
        print(fallback_answer(retrieved))


if __name__ == "__main__":
    main()
