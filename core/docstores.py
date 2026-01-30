from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class Doc:
    id: str
    text: str
    meta: Dict[str, Any]

class BaseDocStore(ABC):
    @abstractmethod
    def get(self, idx: int) -> Doc:
        ...

    def batch_get(self, indices: List[int]) -> List[Doc]:
        return [self.get(i) for i in indices]

class JSONLInMemoryDocStore(BaseDocStore):
    """
    Loads JSONL into RAM once. Fast lookups.

    IMPORTANT INVARIANT:
      - JSONL line number (0-based) == FAISS internal vector id.
    """

    def __init__(self, path: str):
        self.docs: List[Doc] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e

                self.docs.append(
                    Doc(
                        id=str(obj.get("id", len(self.docs))),
                        text=str(obj.get("text", "")),
                        meta={k: v for k, v in obj.items() if k not in {"id", "text"}},
                    )
                )

        if not self.docs:
            raise ValueError(f"Docstore at {path} is empty.")

    def __len__(self) -> int:
        return len(self.docs)

    def get(self, idx: int) -> Doc:
        if idx < 0 or idx >= len(self.docs):
            return Doc(id=str(idx), text="", meta={})
        return self.docs[idx]


def load_docstore(docstore_path: str, docstore_type: str = "jsonl") -> BaseDocStore:
    t = docstore_type.lower()
    if t == "jsonl":
        return JSONLInMemoryDocStore(docstore_path)
    raise ValueError(f"Unknown docstore_type: {docstore_type}")
