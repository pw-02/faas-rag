from faasrag.core.embedders import build_embedder
from faasrag.core.args import EmbedderConfig, DPREmbedderConfig

def test_question_embedder():
    config =DPREmbedderConfig(
            query_encoder_id="sentence-transformers/facebook-dpr-question_encoder-single-nq-base",
            passage_encoder_id="sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base",

    )
    embedder = build_embedder(config, device="cpu")
    q = "who wrote pride and prejudice?"
    vec = embedder.embed_queries([q])[0]
    print(f"Query: {q}")
    print(f"Embedding (first 10 dims): {vec[:10]}")
    print(f"Embedding shape: {vec.shape}")

if __name__ == "__main__":
    test_question_embedder()

