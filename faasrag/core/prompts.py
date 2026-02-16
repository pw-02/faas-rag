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
        f"Question: {q}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_llm_only_messages(question: str) -> List[ChatMessage]:
    q = normalize_question(question)
    return [
        {"role": "system", "content": LLM_ONLY_SYSTEM},
        {"role": "user", "content": q},
    ]


def build_fewshot_messages(passages: List[Passage], max_ctx_chars: int = 4000) -> List[ChatMessage]:
    # Treat passages as the example dialogue text (sanitized)
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
    For LLM_ONLY, passages_used is [].
    """
    if prompt_build_method == PromptBuildMethodType.QA_STRICT:
        messages = build_qa_messages(
            question=question,
            passages=passages,
            max_ctx_chars=max_ctx_chars,
            strict=True,
        )
        passages_used = passages

    elif prompt_build_method == PromptBuildMethodType.QA_OPEN:
        messages = build_qa_messages(
            question=question,
            passages=passages,
            max_ctx_chars=max_ctx_chars,
            strict=False,
        )
        passages_used = passages

    elif prompt_build_method == PromptBuildMethodType.LLM_ONLY:
        messages = build_llm_only_messages(question=question)
        passages_used = []  # ensure no context is used

    elif prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        messages = build_fewshot_messages(passages=passages, max_ctx_chars=max_ctx_chars)
        passages_used = passages

    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    return messages, passages_used
