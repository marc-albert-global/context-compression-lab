"""Metrics: efficiency (token reduction) and information preservation.

Following the evaluation literature (arXiv 2503.19114), preservation is
measured with cheap, offline proxies (entity retention and lexical overlap)
which are useful but, crucially, *not* a guarantee of downstream task accuracy.
That gap is exactly what the optional LLM task-eval (`eval_llm.py`) exposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .compressors.lexical import STOPWORDS
from .tokenization import count_tokens

_WORD = re.compile(r"[A-Za-z]+")
# Proxy "entities": capitalized words (not purely sentence-initial) and numbers.
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+\b|\b\d[\d,.]*\b")


def token_reduction_pct(original: str, compressed: str) -> float:
    o = count_tokens(original)
    if o == 0:
        return 0.0
    return (1 - count_tokens(compressed) / o) * 100


def _content_terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS}


def content_word_jaccard(original: str, compressed: str) -> float:
    a, b = _content_terms(original), _content_terms(compressed)
    if not a:
        return 100.0
    return 100.0 * len(a & b) / len(a | b) if (a | b) else 100.0


def entity_retention_pct(original: str, compressed: str) -> float:
    a = set(_ENTITY.findall(original))
    if not a:
        return 100.0
    b = set(_ENTITY.findall(compressed))
    return 100.0 * len(a & b) / len(a)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(original: str, compressed: str) -> float:
    ref = [w.lower() for w in _WORD.findall(original)]
    cand = [w.lower() for w in _WORD.findall(compressed)]
    if not ref or not cand:
        return 0.0
    lcs = _lcs(ref, cand)
    r, p = lcs / len(ref), lcs / len(cand)
    return 100.0 * (2 * p * r / (p + r)) if (p + r) else 0.0


def embedding_cosine_pct(original: str, compressed: str) -> float | None:
    """Optional: sentence-embedding cosine similarity. Needs the `embed` extra."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return None
    model = _embed_model()
    if model is None:
        return None
    emb = model.encode([original, compressed], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1])) * 100


_EMBED = {}


def _embed_model():
    if "m" not in _EMBED:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBED["m"] = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBED["m"] = None
    return _EMBED["m"]


@dataclass
class Scores:
    token_reduction: float
    entity_retention: float
    rouge_l: float
    content_jaccard: float
    embedding_cosine: float | None = None

    @property
    def preservation(self) -> float:
        """Headline context-preservation %: mean of entity retention and ROUGE-L."""
        return (self.entity_retention + self.rouge_l) / 2


def score(original: str, compressed: str, *, embedding: bool = False) -> Scores:
    return Scores(
        token_reduction=token_reduction_pct(original, compressed),
        entity_retention=entity_retention_pct(original, compressed),
        rouge_l=rouge_l_f1(original, compressed),
        content_jaccard=content_word_jaccard(original, compressed),
        embedding_cosine=embedding_cosine_pct(original, compressed) if embedding else None,
    )
