"""Load the benchmark corpus (passages + QA pairs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "passages.jsonl"


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    passage: str
    question: str
    answer: str


def load(path: Path = CORPUS_PATH) -> list[Item]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(Item(d["id"], d["title"], d["passage"], d["question"], d["answer"]))
    return items
