"""Token counting.

Uses a real BPE tokenizer (`tiktoken`, OpenAI's `cl100k_base`) so that
byte-pair-encoding effects are genuine — most importantly the counterintuitive
result that lemmatization/stemming can *increase* token count by producing word
forms the merges don't cover. If tiktoken can't load (no cached vocab, offline),
we fall back to a GPT-style regex pre-tokenizer so the lab still runs; the
fallback is approximate and labeled as such.
"""

from __future__ import annotations

import re
from functools import lru_cache

# GPT-style pre-tokenization pattern (approximation used only as a fallback).
_FALLBACK_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+"
)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _encoder() is not None else "regex-fallback"


def count_tokens(text: str) -> int:
    """Number of tokens in `text` under the active tokenizer."""
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return sum(1 for _ in _FALLBACK_RE.finditer(text))
