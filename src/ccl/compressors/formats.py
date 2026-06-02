"""Structure / format compressors (hard-prompt, rewriting).

Reformatting is a real lever: the literature and practitioner benchmarks show
Markdown is ~15% cheaper than JSON and TSV ~50% cheaper for tabular data. For
prose, converting to terse bullet points and de-duplicating repeated sentences
removes connective filler while keeping the propositions.
"""

from __future__ import annotations

import re

from .base import Compressor, register
from .extractive import _split_sentences

# Leading connective phrases that add no information at the start of a sentence.
_FILLER_PREFIX = re.compile(
    r"^(however|moreover|furthermore|in addition|additionally|therefore|thus|"
    r"as a result|on the other hand|for example|for instance|in fact|indeed|"
    r"that said|in other words|of course|it should be noted that|note that)[,:]?\s+",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+")


@register
class ProseToBullets(Compressor):
    """One bullet per sentence, with leading connective filler stripped."""

    name = "bullets"
    category = "format/rewrite"

    def compress(self, text: str) -> str:
        bullets = []
        for s in _split_sentences(text):
            s = _FILLER_PREFIX.sub("", s).rstrip(".")
            if s:
                bullets.append("- " + s)
        return "\n".join(bullets) if bullets else text


@register
class DedupeSentences(Compressor):
    """Drop sentences that repeat content already seen (by content-term set)."""

    name = "dedupe"
    category = "format/filtering"

    def compress(self, text: str) -> str:
        seen: set[frozenset] = set()
        kept = []
        for s in _split_sentences(text):
            key = frozenset(w.lower() for w in re.findall(r"[A-Za-z]+", s))
            if key and key in seen:
                continue
            seen.add(key)
            kept.append(s)
        return " ".join(kept)
