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


def build_faiss_index_ivf(
    embed_model: SentenceTransformer,
    docs: List[str],
    chunk_size: int = 240,
    overlap: int = 60,
    nlist: int = 32,
) -> Tuple[faiss.Index, List[Chunk], int]:
    """
    Returns (faiss_index, chunks, embedding_dim)

    IVF notes:
      - Requires training: index.train(train_vectors)
      - index.nprobe controls recall/speed at query time.
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

    # Quantizer defines coarse centroids (also uses IP since we normalized)
    quantizer = faiss.IndexFlatIP(dim)

    # IVF Flat stores full vectors (no compression)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # IVF must be trained before adding vectors
    # For large corpora: train on a random sample (e.g., 50k-200k vectors).
    # For small demo datasets: train on all vectors.
    index.train(xb)
    index.add(xb)

    return index, chunks, dim


def retrieve(
    embed_model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[Chunk],
    query: str,
    top_k: int = 3,
    nprobe: int = 6,
) -> List[Dict[str, Any]]:
    # IVF search-time knob
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

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
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "Retrieval-Augmented Generation (RAG) combines retrieval with text generation to improve factuality.",
        "Embeddings map text to vectors so that semantically similar texts are close in vector space.",
        "Chunking helps retrieval by splitting long documents into smaller passages, improving recall and relevance.",
        "IVF (inverted file) indexes partition vectors into clusters and search only a subset of clusters at query time.",
        "IVF indexes must be trained on sample vectors before you can add data.",
    ]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # For tiny datasets, keep nlist small (rule of thumb: nlist << #vectors)
    index, chunks, dim = build_faiss_index_ivf(
        embed_model, docs, chunk_size=240, overlap=60, nlist=8
    )
    print(f"Built FAISS IVF index with {index.ntotal} vectors (dim={dim}), nlist={index.nlist}.")

    query = "Do IVF indexes need training and why?"
    retrieved = retrieve(embed_model, index, chunks, query, top_k=3, nprobe=4)

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
