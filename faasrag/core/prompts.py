from __future__ import annotations

from typing import List, Dict, Literal, Tuple, Optional
from enum import Enum, auto
import re

from faasrag.core.args import Passage


# ============================================================
# Prompt Modes
# ============================================================

class PromptBuildMethodType(Enum):
    QA_STRICT = auto()
    QA_OPEN = auto()
    LLM_ONLY = auto()
    FEW_SHOT = auto()
    LOGIT_RAG_STAGE1 = auto()  # not a prompt method, but used to trigger LLM_ONLY with top_k>0


# ============================================================
# Types
# ============================================================

Role = Literal["system", "user", "assistant"]
ChatMessage = Dict[str, str]  # {"role": Role, "content": str}


# ============================================================
# System Instructions
# ============================================================

STRICT_SYSTEM = (
    "Answer the question using ONLY the provided passages. "
    "If the answer is not in the passages, reply exactly: I don't know. "
    "Output ONLY the answer (max 5 words)."
)

OPEN_SYSTEM = (
    "Answer the question using the provided passages if relevant, but you may also rely on general knowledge. "
    "Output ONLY the answer (max 5 words)."
)

LLM_ONLY_SYSTEM = (
    "Answer using general knowledge. "
    "Return ONLY the answer. "
    "No extra words. No punctuation. "
    "Maximum 5 words."
)

# This is used for scoring candidates (stage-1). It should be strict about format.
STAGE1_SYSTEM = (
    "You are scoring short-answer candidates for a factual question. "
    "Return ONLY the short answer in the requested format."
)


FEWSHOT_SYSTEM = "Summarize the dialogue into a few short sentences."


# ============================================================
# Helpers
# ============================================================

def normalize_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return "?"
    if not question.endswith("?"):
        question += "?"
    return question[0].lower() + question[1:]


_ROLE_LINE = re.compile(r"(?im)^\s*(system|user|assistant)\s*:\s*")

def sanitize_passage_text(text: str) -> str:
    """
    Helps prevent retrieved text from injecting fake chat roles like 'user:'.
    """
    if not text:
        return ""
    text = text.replace("\0", "")
    text = _ROLE_LINE.sub("", text)
    return text.strip()


def format_context(passages: List[Passage], max_ctx_chars: Optional[int] = None) -> str:
    """
    Concatenate passages with optional global character budget.
    Wrap as data to reduce instruction-following from passages.
    """
    chunks: List[str] = []
    total = 0

    for p in passages:
        title = (p.title or "").strip()
        body = sanitize_passage_text(p.text or "")

        block = f"Title: {title}\n{body}\n"

        if max_ctx_chars is not None:
            if total + len(block) > max_ctx_chars:
                remain = max_ctx_chars - total
                if remain <= 0:
                    break
                block = block[:remain]

        chunks.append(block)
        total += len(block)

    return "\n".join(chunks).strip()


def infer_answer_type(question: str) -> str:
    """
    Very lightweight heuristic. Good enough to stop obvious failures like picking titles for 'how many' questions.
    Returns: one of {"person", "number", "date", "location", "other"}.
    """
    q = normalize_question(question).lower()

    if q.startswith(("who ", "whose ")):
        return "person"

    if q.startswith(("how many ", "how much ")):
        return "number"

    if q.startswith(("when ", "what year ", "what month ", "what date ")):
        return "date"

    if q.startswith(("where ", "what city ", "what country ", "what state ")):
        return "location"

    return "other"



# ============================================================
# Chat Prompt Builders (return messages)
# ============================================================
def build_qa_messages(
    question: str,
    passages: List[Passage],
    max_ctx_chars: int,
    strict: bool,
) -> List[ChatMessage]:
    q = normalize_question(question)
    context = format_context(passages, max_ctx_chars)

    system = STRICT_SYSTEM if strict else OPEN_SYSTEM

    user_content = (
        "Context (data):\n"
        '"""\n'
        f"{context}\n"
        '"""\n\n'
        f"Question: {q}\n"
        "Short answer:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_llm_only_messages(question: str) -> List[ChatMessage]:
    q = normalize_question(question)
    return [
        {"role": "system", "content": LLM_ONLY_SYSTEM},
        {"role": "user", "content": f"{q}\nShort answer:"},
    ]


def build_fewshot_messages(passages: List[Passage], max_ctx_chars: int = 4000) -> List[ChatMessage]:
    examples = []
    total = 0
    for p in passages:
        t = sanitize_passage_text(p.text or "")
        if not t:
            continue
        if total + len(t) > max_ctx_chars:
            remain = max_ctx_chars - total
            if remain <= 0:
                break
            t = t[:remain]
        examples.append(t)
        total += len(t)

    user_content = "\n\n".join(examples).strip()

    return [
        {"role": "system", "content": FEWSHOT_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_stage1_scoring_messages(question: str) -> List[ChatMessage]:
    """
    Messages used when scoring a candidate completion in Stage-1.
    Your generator will compute log P(candidate | these messages).
    """
    q = normalize_question(question)
    atype = infer_answer_type(q)

    if atype == "person":
        slot = "Person"
        extra = "Return ONLY the person name. Do not repeat the question or the title."
    elif atype == "number":
        slot = "Number"
        extra = "Return ONLY the number (digits) if possible. Do not repeat the question or the title."
    elif atype == "date":
        slot = "Date"
        extra = "Return ONLY the date or year. Do not repeat the question or the title."
    elif atype == "location":
        slot = "Location"
        extra = "Return ONLY the location name. Do not repeat the question or the title."
    else:
        slot = "Short answer"
        extra = "Return ONLY the short answer. Do not repeat the question."

    user_content = (
        f"{extra}\n\n"
        f"Question: {q}\n"
        f"{slot}:"
    )

    return [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ============================================================
# Unified Entry Point
# ============================================================

def build_rag_messages(
    question: str,
    passages: List[Passage],
    prompt_build_method: PromptBuildMethodType,
    max_ctx_chars: int = 4000,
) -> Tuple[List[ChatMessage], List[Passage]]:
    """
    Returns (messages, passages_used).
    For LLM_ONLY / LOGIT_RAG_STAGE1, passages_used is [].
    """
    if prompt_build_method == PromptBuildMethodType.QA_STRICT:
        messages = build_qa_messages(question, passages, max_ctx_chars, strict=True)
        passages_used = passages

    elif prompt_build_method == PromptBuildMethodType.QA_OPEN:
        messages = build_qa_messages(question, passages, max_ctx_chars, strict=False)
        passages_used = passages

    elif prompt_build_method == PromptBuildMethodType.LLM_ONLY:
        messages = build_llm_only_messages(question)
        passages_used = []

    elif prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        messages = build_fewshot_messages(passages, max_ctx_chars=max_ctx_chars)
        passages_used = passages

    elif prompt_build_method == PromptBuildMethodType.LOGIT_RAG_STAGE1:
        messages = build_stage1_scoring_messages(question)
        passages_used = []

    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    return messages, passages_used