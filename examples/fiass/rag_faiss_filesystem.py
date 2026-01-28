import os
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunk_meta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        path TEXT NOT NULL
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


def write_chunk_to_disk(chunks_dir: Path, chunk_id: int, text: str) -> str:
    # Store as UTF-8 text file. For speed/space, consider gzip or parquet.
    p = chunks_dir / f"chunk_{chunk_id}.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def build_index_and_store(
    conn: sqlite3.Connection,
    chunks_dir: Path,
    embed_model: SentenceTransformer,
    docs: List[str],
) -> faiss.Index:
    chunks_dir.mkdir(parents=True, exist_ok=True)

    inserted: List[Tuple[int, int, str]] = []  # (chunk_id, doc_id, path)

    # 1) Create metadata rows first, then write files using chunk_id
    for doc_id, doc_text in enumerate(docs):
        for ch in chunk_text(doc_text):
            cur = conn.execute(
                "INSERT INTO chunk_meta (doc_id, path) VALUES (?, ?)",
                (doc_id, ""),  # placeholder; we update after writing
            )
            chunk_id = cur.lastrowid
            path = write_chunk_to_disk(chunks_dir, chunk_id, ch)
            conn.execute("UPDATE chunk_meta SET path = ? WHERE id = ?", (path, chunk_id))
            inserted.append((chunk_id, doc_id, path))
    conn.commit()

    # 2) Embed by reading the chunk files (or embed before writing; either is fine)
    texts = [Path(p).read_text(encoding="utf-8") for (_, _, p) in inserted]
    emb = embed_model.encode(texts, normalize_embeddings=True, batch_size=64)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base)

    ids = np.asarray([cid for (cid, _, _) in inserted], dtype="int64")
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
            "SELECT id, doc_id, path FROM chunk_meta WHERE id = ?",
            (int(chunk_id),),
        ).fetchone()
        if not row:
            continue

        chunk_path = row[2]
        chunk_text = Path(chunk_path).read_text(encoding="utf-8")

        results.append({
            "score": float(score),
            "chunk_id": row[0],
            "doc_id": row[1],
            "path": chunk_path,
            "chunk_text": chunk_text,
        })
    return results


def main():
    docs = [
        "Option 2 stores chunk bodies on disk (filesystem or object storage) and keeps only paths in SQLite.",
        "This keeps the database smaller and can reduce DB IO for large text blobs.",
        "At query time, you fetch metadata rows, then load the chunk content by path and pass it to the LLM.",
        "FAISS stores vectors; you still need a docstore to map returned ids to actual text.",
    ]

    conn = sqlite3.connect("chunk_meta.db")
    init_db(conn)

    chunks_dir = Path("chunk_store")

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    index = build_index_and_store(conn, chunks_dir, embed_model, docs)
    print(f"Indexed {index.ntotal} vectors. (Chunk text stored on disk in {chunks_dir}/)")

    query = "Why store chunks on disk instead of in SQLite?"
    results = retrieve(conn, embed_model, index, query, top_k=3)

    print("\n=== Retrieved ===")
    for r in results:
        print(f"- score={r['score']:.3f} id={r['chunk_id']} doc={r['doc_id']} path={r['path']}")
        print(f"  text: {r['chunk_text']}")

    conn.close()


if __name__ == "__main__":
    main()
