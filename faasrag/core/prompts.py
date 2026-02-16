from __future__ import annotations

from typing import List
from enum import Enum, auto
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
# System Instructions
# ============================================================

STRICT_SYSTEM = (
    "Answer the question using ONLY the provided passages. "
    "If the answer is not in the passages, reply exactly: I don't know. "
    "Output ONLY the answer (max 5 words)."
)


OPEN_SYSTEM = (
    "Answer the question using the provided passages if relevant, but you may also rely on general knowledge. "
    # "Prefer answers grounded in the passages. "
    "Output ONLY the answer (max 5 words)."
)

LLM_ONLY_SYSTEM = (
    "Answer the question using general knowledge. "
    "Do not mention passages or retrieval. "
    "Output ONLY the answer (max 5 words)."
)

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


def format_context(passages: List[Passage], max_ctx_chars: int | None = None) -> str:
    """
    Concatenate passages with optional global character budget.
    """
    chunks = []
    total = 0

    for p in passages:
        block = f"Title: {p.title}\n{p.text}\n"

        if max_ctx_chars:
            if total + len(block) > max_ctx_chars:
                remain = max_ctx_chars - total
                if remain <= 0:
                    break
                block = block[:remain]

        chunks.append(block)
        total += len(block)

    return "\n".join(chunks)


# ============================================================
# QA Prompt Builder
# ============================================================

def build_qa_prompt(
    question: str,
    passages: List[Passage],
    max_ctx_chars: int,
    strict: bool,
) -> str:
    q = normalize_question(question)
    context = format_context(passages, max_ctx_chars)

    system = STRICT_SYSTEM if strict else OPEN_SYSTEM

    return (
        "system:\n"
        f"{system}\n\n"
        "user:\n"
        f"Context:\n{context}\n\n"
        f"Question: {q}\n\n"
        "assistant:"
    )


# ============================================================
# FEW SHOT Prompt Builder
# ============================================================

def build_fewshot_prompt(passages: List[Passage]) -> str:
    examples = "\n".join(p.text for p in passages)

    return (
        "system:\n"
        "Summarize the dialogue into a few short sentences.\n\n"
        "user:\n"
        f"{examples}\n\n"
        "assistant:"
    )


def build_llm_only_prompt(question: str) -> str:
    q = normalize_question(question)
    return (
        "system:\n"
        f"{LLM_ONLY_SYSTEM}\n\n"
        "user:\n"
        f"Question: {q}\n\n"
        "assistant:"
    )


# ============================================================
# Unified Entry Point
# ============================================================

def build_rag_prompt(
    question: str,
    passages: List[Passage],
    prompt_build_method: PromptBuildMethodType,
    max_ctx_chars: int = 4000,
):
    if prompt_build_method == PromptBuildMethodType.QA_STRICT:
        prompt = build_qa_prompt(
            question=question,
            passages=passages,
            max_ctx_chars=max_ctx_chars,
            strict=True,
        )

    elif prompt_build_method == PromptBuildMethodType.QA_OPEN:
        prompt = build_qa_prompt(
            question=question,
            passages=passages,
            max_ctx_chars=max_ctx_chars,
            strict=False,
        )

    elif prompt_build_method == PromptBuildMethodType.LLM_ONLY:
        prompt = build_llm_only_prompt(question=question)
        passages = []  # important: ensure no context is passed along

    elif prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        prompt = build_fewshot_prompt(passages)

    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    return prompt, passages


