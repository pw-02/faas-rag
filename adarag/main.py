
import re
# pip install transformers accelerate torch faiss-cpu numpy scikit-learn tqdm

import os
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import faiss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

STOP = 0
RETRIEVE = 1

# ----------------------------
# Text normalization + metrics
# ----------------------------

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    from collections import Counter
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)



@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float

class STRetriever:
    def __init__(
        self,
        faiss_index_path: str,
        passage_store_path: str,
        question_encoder_name_or_path: str,
        device: str = "cuda",
    ):
        self.index = faiss.read_index(faiss_index_path)

        with open(passage_store_path, "r", encoding="utf-8") as f:
            self.passages: Dict[str, Any] = json.load(f)

        self.qenc = SentenceTransformer(question_encoder_name_or_path, device=device)

        # sanity check dimension
        test_vec = self.encode_question("dimension check")
        if test_vec.shape[0] != self.index.d:
            raise ValueError(
                f"Embedding dim {test_vec.shape[0]} != FAISS index dim {self.index.d}"
            )

    def encode_question(self, question: str) -> np.ndarray:
        vec = self.qenc.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=False,   # key change: DO NOT normalize
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def retrieve(self, question: str, k: int = 5) -> List[Passage]:
        qvec = self.encode_question(question).reshape(1, -1)
        scores, idxs = self.index.search(qvec, k)

        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out = []
        for score, pid in zip(scores, idxs):
            meta = self.passages.get(str(pid))
            if meta is None:
                continue
            title = meta.get("title", "") or ""
            text = meta.get("text", "") or meta.get("contents", "") or ""
            out.append(Passage(pid=int(pid), title=title, text=text, score=float(score)))
        return out


class LocalGenerator:
    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 64):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map="auto" if device.startswith("cuda") else None
        )
        self.model.eval()

    @torch.no_grad()
    def generate_with_entropy(self, prompt: str, temperature: float = 0.0) -> Dict:
        """
        temperature=0 => greedy. If >0, uses sampling (entropy computation still ok).
        Returns dict with: text, gen_tokens, mean_entropy.
        """
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        input_ids = inputs["input_ids"]

        # We'll do token-by-token generation to compute entropy from logits.
        generated = input_ids
        entropies = []

        for _ in range(self.max_new_tokens):
            out = self.model(input_ids=generated)
            logits = out.logits[:, -1, :]  # (1, vocab)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)  # (1,)
            entropies.append(entropy.item())

            if temperature == 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / max(temperature, 1e-6)
                probs_s = F.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs_s, num_samples=1)

            generated = torch.cat([generated, next_id], dim=-1)

            if next_id.item() == self.tok.eos_token_id:
                break

        gen_tokens = generated.shape[-1] - input_ids.shape[-1]
        text = self.tok.decode(generated[0][input_ids.shape[-1]:], skip_special_tokens=True)

        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        return {"text": text.strip(), "gen_tokens": int(gen_tokens), "mean_entropy": mean_entropy}
    
    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[str, int]:
        inputs = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # greedy baseline
            temperature=0.0,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return text, int(gen_ids.shape[-1])



def format_context(passages: List[Passage], max_chars: int = 4000) -> str:
    ctx = ""
    for p in passages:
        block = f"[{p.pid}] {p.title}\n{p.text}\n\n"
        if len(ctx) + len(block) > max_chars:
            break
        ctx += block
    return ctx.strip()

def build_prompt(question: str, passages: Optional[List[Passage]] = None) -> str:
    if passages:
        ctx = format_context(passages)
        return (
            "Answer the question using ONLY the context. If the answer is not in the context, say \"I don't know\".\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {question}\nAnswer:"
        )
    else:
        return (
            "Answer the question. If you are not sure, say \"I don't know\".\n\n"
            f"Question: {question}\nAnswer:"
        )



def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

@dataclass
class PlanResult:
    final_answer: str
    correctness: float
    retrieval_calls: int
    retrieved_chars: int
    generated_tokens: int
    states_actions: List[Tuple[np.ndarray, int]]  # (state_features, action_label)
    debug: Dict

def reward(correctness: float, retrieval_calls: int, retrieved_chars: int, generated_tokens: int,
           lam_call=0.20, lam_ctx=0.00002, lam_gen=0.002) -> float:
    """
    Tune these weights. They control "accuracy vs cost".
    retrieved_chars is a proxy for retrieved tokens. Replace with actual token count if you prefer.
    """
    return correctness - lam_call*retrieval_calls - lam_ctx*retrieved_chars - lam_gen*generated_tokens


def build_state_features(step: int,
                         mean_entropy: float,
                         top_score: float,
                         score_gap: float,
                         ctx_chars: int,
                         retrieval_calls: int) -> np.ndarray:
    return np.array([step, mean_entropy, top_score, score_gap, ctx_chars, retrieval_calls], dtype=np.float32)



def run_plan(question: str, gold: str,
             plan: str,
             retriever: STRetriever,
             gen: LocalGenerator,
             max_ctx_chars: int = 4000) -> PlanResult:

    retrieval_calls = 0
    retrieved_chars = 0
    generated_tokens = 0
    debug = {"plan": plan}

    # Step 0: draft answer (no context) to compute uncertainty state
    draft = gen.generate_with_entropy(build_prompt(question, None), temperature=0.0)
    generated_tokens += draft["gen_tokens"]

    # No retrieval stats yet
    top_score_0 = 0.0
    score_gap_0 = 0.0
    ctx_chars_0 = 0
    s0 = build_state_features(step=0,
                              mean_entropy=draft["mean_entropy"],
                              top_score=top_score_0,
                              score_gap=score_gap_0,
                              ctx_chars=ctx_chars_0,
                              retrieval_calls=retrieval_calls)

    states_actions = []

    if plan == "NR":
        # Oracle action at step 0 is STOP
        states_actions.append((s0, STOP))
        final_answer = draft["text"]

        corr = exact_match(final_answer, gold)
        return PlanResult(final_answer, corr, retrieval_calls, retrieved_chars, generated_tokens, states_actions, debug)

    # Otherwise step 0 action is RETRIEVE
    states_actions.append((s0, RETRIEVE))

    def retrieve_once(q: str, k: int) -> List[Passage]:
        nonlocal retrieval_calls, retrieved_chars
        retrieval_calls += 1
        ps = retriever.retrieve(q, k=k)
        # track rough context size
        retrieved_chars += sum(len(p.text) for p in ps)
        return ps

    # Round 1 retrieval
    k1 = 5 if plan in ("R5", "R5_R5") else 20
    p1 = retrieve_once(question, k1)

    # Generate answer with evidence
    out1 = gen.generate_with_entropy(build_prompt(question, p1), temperature=0.0)
    generated_tokens += out1["gen_tokens"]

    # Build state after round 1
    top_score_1 = float(p1[0].score) if p1 else 0.0
    score_gap_1 = float(p1[0].score - p1[1].score) if len(p1) > 1 else 0.0
    ctx_chars_1 = len(format_context(p1, max_chars=max_ctx_chars))
    s1 = build_state_features(step=1,
                              mean_entropy=out1["mean_entropy"],
                              top_score=top_score_1,
                              score_gap=score_gap_1,
                              ctx_chars=ctx_chars_1,
                              retrieval_calls=retrieval_calls)

    if plan in ("R5", "R20"):
        # step 1 action is STOP
        states_actions.append((s1, STOP))
        final_answer = out1["text"]
        corr = exact_match(final_answer, gold)
        return PlanResult(final_answer, corr, retrieval_calls, retrieved_chars, generated_tokens, states_actions, debug)

    # plan == "R5_R5": decide to retrieve again then answer
    states_actions.append((s1, RETRIEVE))

    # Round 2 retrieval: simplest is reuse the original question (upgrade later with rewrite)
    p2 = retrieve_once(question, 5)

    # Merge contexts: keep top from both, or just concatenate then truncate by max_ctx_chars
    merged = p1 + p2
    # de-dup by pid
    seen = set()
    merged_unique = []
    for p in merged:
        if p.pid not in seen:
            merged_unique.append(p)
            seen.add(p.pid)

    out2 = gen.generate_with_entropy(build_prompt(question, merged_unique), temperature=0.0)
    generated_tokens += out2["gen_tokens"]

    top_score_2 = float(merged_unique[0].score) if merged_unique else 0.0
    score_gap_2 = float(merged_unique[0].score - merged_unique[1].score) if len(merged_unique) > 1 else 0.0
    ctx_chars_2 = len(format_context(merged_unique, max_chars=max_ctx_chars))
    s2 = build_state_features(step=2,
                              mean_entropy=out2["mean_entropy"],
                              top_score=top_score_2,
                              score_gap=score_gap_2,
                              ctx_chars=ctx_chars_2,
                              retrieval_calls=retrieval_calls)

    states_actions.append((s2, STOP))
    final_answer = out2["text"]
    corr = exact_match(final_answer, gold)
    return PlanResult(final_answer, corr, retrieval_calls, retrieved_chars, generated_tokens, states_actions, debug)





def build_imitation_dataset(train_examples: List[Dict],
                            retriever: STRetriever,
                            gen: LocalGenerator,
                            plans=("NR", "R5", "R20", "R5_R5"),
                            lam_call=0.20, lam_ctx=0.00002, lam_gen=0.002):

    X = []
    y = []
    chosen = []

    for ex in tqdm(train_examples, desc="Oracle simulation"):
        q = ex["question"]
        gold = ex["answer"]

        best = None
        best_r = -1e9

        for plan in plans:
            res = run_plan(q, gold, plan, retriever, gen)
            r = reward(res.correctness, res.retrieval_calls, res.retrieved_chars, res.generated_tokens,
                       lam_call=lam_call, lam_ctx=lam_ctx, lam_gen=lam_gen)
            if r > best_r:
                best_r = r
                best = res

        # convert best trajectory into supervised examples
        for state_vec, action in best.states_actions:
            X.append(state_vec)
            y.append(action)

        chosen.append({"question": q, "best_plan": best.debug["plan"], "reward": best_r,
                       "correct": best.correctness, "retrieval_calls": best.retrieval_calls})

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, chosen




def train_controller(X: np.ndarray, y: np.ndarray):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    print("Controller accuracy:", accuracy_score(yte, pred))
    return clf


def adaptive_answer(question: str,
                    retriever: STRetriever,
                    gen: LocalGenerator,
                    controller,
                    max_steps: int = 2,
                    k: int = 5,
                    max_ctx_chars: int = 4000):

    retrieval_calls = 0
    passages = []
    ctx_chars = 0

    # Step 0: draft
    draft = gen.generate_with_entropy(build_prompt(question, None), temperature=0.0)
    s = build_state_features(step=0,
                             mean_entropy=draft["mean_entropy"],
                             top_score=0.0,
                             score_gap=0.0,
                             ctx_chars=0,
                             retrieval_calls=0).reshape(1, -1)
    a = int(controller.predict(s)[0])

    if a == STOP:
        return draft["text"]

    cur_answer = draft["text"]
    for step in range(1, max_steps + 1):
        retrieval_calls += 1
        new_ps = retriever.retrieve(question, k=k)  # upgrade later: query rewrite / use cur_answer
        # merge de-dup
        seen = set(p.pid for p in passages)
        for p in new_ps:
            if p.pid not in seen:
                passages.append(p)
                seen.add(p.pid)

        ctx = format_context(passages, max_chars=max_ctx_chars)
        ctx_chars = len(ctx)

        out = gen.generate_with_entropy(build_prompt(question, passages), temperature=0.0)
        cur_answer = out["text"]

        top_score = float(passages[0].score) if passages else 0.0
        score_gap = float(passages[0].score - passages[1].score) if len(passages) > 1 else 0.0

        s = build_state_features(step=step,
                                 mean_entropy=out["mean_entropy"],
                                 top_score=top_score,
                                 score_gap=score_gap,
                                 ctx_chars=ctx_chars,
                                 retrieval_calls=retrieval_calls).reshape(1, -1)
        a = int(controller.predict(s)[0])
        if a == STOP:
            break

    return cur_answer



def load_dataset(path: str, limit: int = 2000) -> List[Dict]:
    """
    Expects jsonl with {"question":..., "answer":...}
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data.append(json.loads(line))
    return data



def eval_rag_baseline(examples: List[Dict[str, str]], retriever: STRetriever, generator: LocalGenerator, k: int = 5,
                      max_ctx_chars: int = 4000, show_examples: int = 5,
                      ) -> Dict[str, Any]:

    ems = []
    f1s = []
    total_gen_tokens = 0

    printed = 0
    outputs = []
    for ex in tqdm(examples, desc=f"Baseline RAG k={k}"):
        q = ex["question"]
        gold = ex["answer"]

        passages = retriever.retrieve(q, k=k)
        prompt = build_prompt(q, passages, max_ctx_chars=max_ctx_chars)
        pred, gen_tokens = generator.generate(prompt)

        total_gen_tokens += gen_tokens
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        ems.append(em)
        f1s.append(f1)

        outputs.append({
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "top_passages": [
                {"pid": p.pid, "score": p.score, "title": p.title}
                for p in passages[:3]
            ]
        })

        if printed < show_examples:
            printed += 1
            print("\n" + "=" * 90)
            print("Q:", q)
            print("Gold:", gold)
            print("Pred:", pred)
            print(f"EM={em:.0f}  F1={f1:.3f}  gen_tokens={gen_tokens}")
            print("\nTop retrieved passages:")
            for p in passages[:3]:
                snippet = (p.text[:240] + "...") if len(p.text) > 240 else p.text
                print(f"  pid={p.pid} score={p.score:.4f} title={p.title}")
                print(f"    {snippet.replace('\\n',' ')}")

    results = {
        "N": len(examples),
        "EM": float(np.mean(ems)) if ems else 0.0,
        "F1": float(np.mean(f1s)) if f1s else 0.0,
        "avg_gen_tokens": float(total_gen_tokens / max(1, len(examples))),
        "outputs": outputs,
    }
    return results




def main():
    # Paths you must set
    faiss_index_path = "/path/to/wiki_dpr.index"
    passage_store_path = "/path/to/wiki_passages.json"
    train_path = "/path/to/train.jsonl"
    test_path = "/path/to/dev.jsonl"

    # Models you must set
    generator_model = "meta-llama/Meta-Llama-3-8B-Instruct"  # example; use what you have locally

    retriever = STRetriever(faiss_index_path, passage_store_path)
    gen = LocalGenerator(generator_model, device="cuda", max_new_tokens=48)

    train_data = load_dataset(train_path, limit=500)  # start small
    X, y, chosen = build_imitation_dataset(train_data, retriever, gen)

    controller = train_controller(X, y)

    # quick eval
    dev_data = load_dataset(test_path, limit=100)
    correct = 0
    for ex in tqdm(dev_data, desc="Adaptive eval"):
        pred = adaptive_answer(ex["question"], retriever, gen, controller)
        correct += int(exact_match(pred, ex["answer"]))
    print("Adaptive EM:", correct / len(dev_data))

if __name__ == "__main__":
    main()
