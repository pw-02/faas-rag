import sqlite3
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL
    )
    """)
    conn.commit()


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


def build_index_and_store(
    conn: sqlite3.Connection,
    embed_model: SentenceTransformer,
    docs: List[str],
) -> faiss.Index:
    """
    Inserts chunk text into SQLite. Uses the SQLite row id as the FAISS vector id.
    """
    # 1) Insert all chunks, keep their assigned ids (primary keys)
    chunk_rows: List[Tuple[int, int, str]] = []  # (id, doc_id, text)
    for doc_id, doc_text in enumerate(docs):
        for ch in chunk_text(doc_text):
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, chunk_text) VALUES (?, ?)",
                (doc_id, ch),
            )
            chunk_id = cur.lastrowid
            chunk_rows.append((chunk_id, doc_id, ch))
    conn.commit()

    # 2) Embed in batches (important for large data)
    texts = [r[2] for r in chunk_rows]
    emb = embed_model.encode(texts, normalize_embeddings=True, batch_size=64)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]

    # 3) Use IndexIDMap so we can attach our own ids (SQLite primary keys)
    base = faiss.IndexFlatIP(dim)  # cosine if normalized
    index = faiss.IndexIDMap(base)

    ids = np.asarray([r[0] for r in chunk_rows], dtype="int64")
    index.add_with_ids(emb, ids)

    return index


def retrieve(
    conn: sqlite3.Connection,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    query: str,
    top_k: int = 3,
):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:
            continue
        row = conn.execute(
            "SELECT id, doc_id, chunk_text FROM chunks WHERE id = ?",
            (int(chunk_id),),
        ).fetchone()
        if row:
            results.append({
                "score": float(score),
                "chunk_id": row[0],
                "doc_id": row[1],
                "chunk_text": row[2],
            })
    return results


def main():
    # Demo documents (replace with your corpus)
    docs = [
        "FAISS is a library for efficient similarity search over dense vectors.",
        "RAG combines retrieval with generation: retrieve relevant chunks, then generate an answer grounded in them.",
        "Storing chunk text in SQLite keeps Python memory use low for large corpora.",
        "IndexIDMap lets you store your own integer ids in FAISS (e.g., DB primary keys).",
    ]

    conn = sqlite3.connect("chunks.db")
    init_db(conn)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index = build_index_and_store(conn, embed_model, docs)
    print(f"Indexed {index.ntotal} vectors. (Chunk text stored in SQLite.)")

    query = "How does RAG work and why use a database for chunks?"
    results = retrieve(conn, embed_model, index, query, top_k=3)

    print("\n=== Retrieved ===")
    for r in results:
        print(f"- score={r['score']:.3f} id={r['chunk_id']} doc={r['doc_id']}: {r['chunk_text']}")

    conn.close()


if __name__ == "__main__":
    main()
