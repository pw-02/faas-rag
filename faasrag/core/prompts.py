from __future__ import annotations

from typing import Callable, List

from faasrag.core.args import Passage


# Public type for prompt strategies
PromptFn = Callable[[str, list[Passage] | None, int | None], list[dict[str, str]]]

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

def prompt_llm_only(question: str, passages=None, max_ctx_chars=None):
    """
    Pure LLM mode (no retrieval).
    Uses model parametric knowledge.
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


def prompt_rag_strict(question: str, passages: List[Passage], max_ctx_chars: int):
    """
    Strict RAG:
    - MUST use provided context
    - MUST say 'I don't know' if answer not present
    (best for evaluation / hallucination control)
    """
    context = format_context(passages, max_ctx_chars)

    return [
        {
            "role": "system",
            "content": (
                "You must answer ONLY using the provided context. "
                "If the answer is not in the context, say 'I don't know'. "
                "Output a short answer only."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

def prompt_rag_open(question: str, passages: List[Passage], max_ctx_chars: int):
    """
    Open RAG:
    - Use retrieved context if helpful
    - May fall back to general world knowledge
    (recommended default for real systems)
    """
    context = format_context(passages or [], max_ctx_chars or 0)

    return [
        {
            "role": "system",
            "content": (
                "Use the provided context if relevant, but you may also rely on "
                "your general knowledge. Output a short answer only."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]


# def prompt_with_context(question: str, passages: List[Passage], max_ctx_chars: int):
#     """
#     Standard RAG prompt using retrieved passages.
#     """
#     context = format_context(passages, max_ctx_chars)

#     return [
#         {
#             "role": "system",
#             "content": "You answer questions using the provided context. Output a short answer only.",
#         },
#         {
#             "role": "user",
#             "content": (
#                 'Use ONLY this context. Answer with ONLY the answer. '
#                 'If unsure, say "I don\'t know".\n\n'
#                 f"Context:\n{context}\n\n"
#                 f"Question: {question}"
#             ),
#         },
#     ]


# -------------------------
# Registry
# -------------------------

def get_prompt_strategy(name: str) -> PromptFn:
    """
    Resolve prompt strategy by name.

    Supported:
      - llm_only
      - rag_strict
      - rag_open

    Aliases:
      - no_retrieval -> llm_only
      - with_context -> rag_strict
    """
    key = name.lower().replace("-", "_")

    if key in {"llm_only", "no_retrieval"}:
        return prompt_llm_only

    if key in {"rag_strict", "with_context"}:
        return prompt_rag_strict

    if key == "rag_open":
        return prompt_rag_open

    raise ValueError(f"Unknown prompt strategy: {name}")


