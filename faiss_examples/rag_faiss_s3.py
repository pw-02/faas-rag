import os
import sqlite3
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss
import boto3
from sentence_transformers import SentenceTransformer


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS chunk_meta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        bucket TEXT NOT NULL,
        s3_key TEXT NOT NULL
    )
    """)
    conn.commit()


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    # simple character chunking for runnable demo
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


def put_chunk_to_s3(s3, bucket: str, key: str, text: str) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def get_chunk_from_s3(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def build_index_and_store(
    conn: sqlite3.Connection,
    s3,
    bucket: str,
    prefix: str,
    embed_model: SentenceTransformer,
    docs: List[str],
    batch_size: int = 64,
) -> faiss.Index:
    """
    Ingest:
      - chunk doc
      - insert metadata row => chunk_id
      - upload chunk to s3 using chunk_id in key
      - embed chunk text
      - add embedding to FAISS with id=chunk_id (IndexIDMap)
    """
    inserted: List[Tuple[int, int, str, str, str]] = []  # (chunk_id, doc_id, bucket, s3_key, chunk_text)

    for doc_id, doc_text in enumerate(docs):
        for part_i, ch in enumerate(chunk_text(doc_text)):
            # 1) Insert row to get chunk_id
            cur = conn.execute(
                "INSERT INTO chunk_meta (doc_id, bucket, s3_key) VALUES (?, ?, ?)",
                (doc_id, bucket, ""),  # placeholder key; update after we compute it
            )
            chunk_id = cur.lastrowid

            # 2) Create deterministic S3 key (using chunk_id is simplest)
            s3_key = f"{prefix.rstrip('/')}/doc_{doc_id}/chunk_{chunk_id}.txt"

            # 3) Upload chunk body to S3
            put_chunk_to_s3(s3, bucket, s3_key, ch)

            # 4) Update metadata with actual key
            conn.execute(
                "UPDATE chunk_meta SET s3_key = ? WHERE id = ?",
                (s3_key, chunk_id),
            )

            inserted.append((chunk_id, doc_id, bucket, s3_key, ch))

    conn.commit()

    # Embed in batches (for big data, you’d stream this instead of keeping all texts)
    texts = [row[4] for row in inserted]
    emb = embed_model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    base = faiss.IndexFlatIP(dim)       # cosine similarity if embeddings are normalized
    index = faiss.IndexIDMap(base)      # lets us use chunk_id as the vector id

    ids = np.asarray([row[0] for row in inserted], dtype="int64")
    index.add_with_ids(emb, ids)

    return index


def retrieve(
    conn: sqlite3.Connection,
    s3,
    embed_model: SentenceTransformer,
    index: faiss.Index,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results: List[Dict[str, Any]] = []
    for score, chunk_id in zip(scores[0], ids[0]):
        if chunk_id == -1:
            continue

        row = conn.execute(
            "SELECT id, doc_id, bucket, s3_key FROM chunk_meta WHERE id = ?",
            (int(chunk_id),),
        ).fetchone()
        if not row:
            continue

        _, doc_id, bucket, s3_key = row
        chunk_text = get_chunk_from_s3(s3, bucket, s3_key)

        results.append({
            "score": float(score),
            "chunk_id": int(chunk_id),
            "doc_id": int(doc_id),
            "bucket": bucket,
            "s3_key": s3_key,
            "chunk_text": chunk_text,
        })

    return results


def main():
    # ---- configure these ----
    BUCKET = os.environ.get("RAG_S3_BUCKET", "")
    PREFIX = os.environ.get("RAG_S3_PREFIX", "rag_chunks_demo")
    if not BUCKET:
        raise SystemExit("Set RAG_S3_BUCKET env var to your bucket name (e.g., export RAG_S3_BUCKET=my-bucket).")

    # ---- demo docs (replace with your ingestion pipeline) ----
    docs = [
        "FAISS stores vectors; for large datasets you keep chunk text outside of Python memory.",
        "A common scalable pattern is FAISS + docstore. Here docstore is S3 for chunk bodies and SQLite for metadata.",
        "At query time: embed query -> FAISS top-k ids -> lookup ids in SQLite -> fetch text from S3 -> build prompt.",
    ]

    # clients
    s3 = boto3.client("s3")
    conn = sqlite3.connect("chunk_meta.db")
    init_db(conn)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # build index + store chunks in S3
    index = build_index_and_store(conn, s3, BUCKET, PREFIX, embed_model, docs)
    print(f"Indexed {index.ntotal} vectors. Chunk bodies uploaded to s3://{BUCKET}/{PREFIX}/...")

    # query
    query = "How does FAISS + S3 docstore RAG work?"
    results = retrieve(conn, s3, embed_model, index, query, top_k=3)

    print("\n=== Retrieved ===")
    for r in results:
        print(f"- score={r['score']:.3f} id={r['chunk_id']} doc={r['doc_id']} s3://{r['bucket']}/{r['s3_key']}")
        print(f"  text: {r['chunk_text']}")

    conn.close()


if __name__ == "__main__":
    main()
