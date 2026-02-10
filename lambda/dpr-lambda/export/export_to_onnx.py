import os
import torch
from sentence_transformers import SentenceTransformer

# ST_MODEL_ID = "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base"

ST_MODEL_ID = "sentence-transformers/facebook-dpr-question_encoder-single-nq-base"
OUT_DIR = "lambda/dpr-lambda/app/models"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "509"))  # 509 is max for DPR models, but you can set lower for faster inference

class STWrapper(torch.nn.Module):
    def __init__(self, st_model):
        super().__init__()
        self.st = st_model.eval()

    def forward(self, input_ids, attention_mask, token_type_ids):
        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        out = self.st(features)
        return out["sentence_embedding"]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    st = SentenceTransformer(ST_MODEL_ID, device="cpu")
    wrapper = STWrapper(st)

    # ✅ token tensors for ONNX export
    tok = st.tokenizer
    dummy = tok(
        ["hello world"],
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    token_type_ids = dummy.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(dummy["input_ids"])

    onnx_path = os.path.join(OUT_DIR, "model.onnx")

    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"], token_type_ids),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["sentence_embedding_out"],  # ✅ avoid "embedding"
        opset_version=17,
    )

    tok.save_pretrained(OUT_DIR)

    print("Wrote:", onnx_path)

if __name__ == "__main__":
    main()
