import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_name = "BAAI/bge-large-en"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

seq ="Welcome to High Commission of India, Colombo, Sri Lanka Affairs Minister Smt. Sushma Swaraj led an inter-ministerial delegation to Colombo from 5-6 February 2016 for the 9th Session of the India-Sri Lanka Joint Commission. Earlier, External Affairs Minister Smt. Sushma Swaraj was in Colombo on 6-7 March 2015 to prepare for Prime Minister\u2019s visit. EAM visited from 31 August-01 September 2017 to attend the second Indian Ocean Conference organized in Colombo. Commerce and Industry Minister Smt. Nirmala Sitharaman visited Sri Lanka on 26-27 September 2016. Shri Ravi Shankar Prasad, Minister of Law and Justice and Electronics & Information Technology visited Sri Lanka from 14-17 January, 2018. A MoU for"

inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=768)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    out = model(**inputs)
    # Mean pooling
    last_hidden = out.last_hidden_state              # (1, L, H)
    mask = inputs["attention_mask"].unsqueeze(-1)    # (1, L, 1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (1, H)

    emb = F.normalize(pooled, p=2, dim=1)            # (1, H)

print("embedding shape:", emb.shape)  # H is the embedding dim
