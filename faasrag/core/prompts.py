from __future__ import annotations

from typing import Callable, List

from faasrag.core.args import Passage


# -------------------------
# Utilities
# -------------------------

def extract_short_answer(text: str) -> str:
    """
    Normalize model output for QA-style evaluation:
    - first line
    - first sentence
    - strip quotes
    """
    t = text.strip()

    # first line
    t = t.split("\n")[0].strip()

    # first sentence
    t = t.split(".")[0].strip()

    # clean quotes
    t = t.strip('"').strip("'").strip()

    return t


def format_context(passages: List[Passage], max_chars: int) -> str:
    """
    Format retrieved passages into a single context block,
    respecting a max character budget.
    """
    ctx = ""
    for p in passages:
        block = f"[{p.pid}] {p.title}\n{p.text}\n\n"
        if len(ctx) + len(block) > max_chars:
            break
        ctx += block
    return ctx.strip()


# -------------------------
# Prompt strategies
# -------------------------

def prompt_no_retrieval(question: str, passages=None, max_ctx_chars=None):
    """
    Baseline LLM-only prompt (no retrieval).
    """
    return [
        {"role": "system", "content": "You answer questions with a short answer only."},
        {
            "role": "user",
            "content": (
                'Answer with ONLY the answer. If unsure, say "I don\'t know".\n\n'
                f"Question: {question}"
            ),
        },
    ]


def prompt_with_context(question: str, passages: List[Passage], max_ctx_chars: int):
    """
    Standard RAG prompt using retrieved passages.
    """
    context = format_context(passages, max_ctx_chars)

    return [
        {
            "role": "system",
            "content": "You answer questions using the provided context. Output a short answer only.",
        },
        {
            "role": "user",
            "content": (
                'Use ONLY this context. Answer with ONLY the answer. '
                'If unsure, say "I don\'t know".\n\n'
                f"Context:\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]


# -------------------------
# Registry
# -------------------------

def get_prompt_strategy(name: str) -> Callable[..., List[dict[str, str]]]:
    name = name.lower()
    if name == "no_retrieval":
        return prompt_no_retrieval
    elif name == "with_context":
        return prompt_with_context
    else:
        raise ValueError(f"Unknown prompt strategy: {name}")


