import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1) Your documents (toy data)
# ----------------------------
docs = [
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Retrieval-Augmented Generation (RAG) combines retrieval with text generation to improve factuality.",
    "Embeddings map text to vectors so that semantically similar texts are close in vector space.",
    "Chunking helps retrieval by splitting long documents into smaller passages."
]

# ----------------------------
# 2) Chunking (simple example)
#    In real use, chunk by tokens/paragraphs with overlap
# ----------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    # naive character-based chunking for demo purposes
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

chunks = []
chunk_sources = []  # track which doc each chunk came from
for i, d in enumerate(docs):
    for c in chunk_text(d):
        chunks.append(c)
        chunk_sources.append(i)

# ----------------------------
# 3) Embed chunks
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_emb = embed_model.encode(chunks, normalize_embeddings=True)  # (N, dim)
chunk_emb = np.asarray(chunk_emb, dtype="float32")

# ----------------------------
# 4) Build FAISS index
#    Using Inner Product on normalized vectors = cosine similarity
# ----------------------------
dim = chunk_emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_emb)

# ----------------------------
# 5) Retrieval function
# ----------------------------
def retrieve(query: str, top_k: int = 3):
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "score": float(score),
            "chunk": chunks[idx],
            "source_doc_id": chunk_sources[idx],
        })
    return results

# ----------------------------
# 6) Build a prompt from retrieved context
# ----------------------------
def build_prompt(query: str, retrieved):
    context = "\n\n".join([f"[{i}] {r['chunk']}" for i, r in enumerate(retrieved)])
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.

Context:
{context}

Question: {query}

Answer (grounded in context):"""
    return prompt

# ----------------------------
# 7) Optional: call an LLM (OpenAI API example)
# ----------------------------
def answer_with_openai(prompt: str) -> str:
    # Requires: export OPENAI_API_KEY="..."
    from openai import OpenAI
    client = OpenAI()

    # Replace model name with what you have access to in your account
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return resp.output_text

# ----------------------------
# Demo run
# ----------------------------
if __name__ == "__main__":
    query = "What is RAG and why is chunking useful?"
    retrieved = retrieve(query, top_k=3)

    print("=== Retrieved ===")
    for r in retrieved:
        print(f"- score={r['score']:.3f} doc={r['source_doc_id']} chunk={r['chunk']}")

    prompt = build_prompt(query, retrieved)
    print("\n=== Prompt to LLM ===")
    print(prompt)

    # Option A: LLM call
    if os.getenv("OPENAI_API_KEY"):
        print("\n=== LLM Answer ===")
        print(answer_with_openai(prompt))
    else:
        # Option B: no-LLM fallback (just return context)
        print("\n(OPENAI_API_KEY not set — returning retrieved context only.)")
