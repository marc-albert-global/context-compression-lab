"""Lexical / NLP-preprocessing compressors (hard-prompt, surface level)."""

from __future__ import annotations

import re

from .base import Compressor, register

# A compact English stopword list (function words that carry little standalone
# information). Removing these is the one classic preprocessing step that
# reliably reduces BPE token count.
STOPWORDS = frozenset(
    """
    a an the and or but if then else of in on at to from by for with without
    into onto over under again further is are was were be been being am do does
    did doing have has had having i you he she it we they me him her them my your
    his its our their this that these those there here as so than too very can
    will just not no nor only own same such up down out off above below which who
    whom whose what when where why how all any both each few more most other some
    about against between through during before after
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+")
_WS_RE = re.compile(r"\s+")


@register
class Identity(Compressor):
    """Baseline: no compression. Anchors the reduction/preservation scale."""

    name = "identity"
    category = "baseline"

    def compress(self, text: str) -> str:
        return text


@register
class WhitespaceCollapse(Compressor):
    """Collapse runs of whitespace to single spaces and trim."""

    name = "whitespace"
    category = "lexical/normalization"

    def compress(self, text: str) -> str:
        return _WS_RE.sub(" ", text).strip()


@register
class PunctuationStrip(Compressor):
    """Drop most punctuation, keeping sentence-ending marks and digits intact."""

    name = "punctuation"
    category = "lexical/normalization"

    def compress(self, text: str) -> str:
        kept = re.sub(r"[^\w\s.?!]", "", text)
        return _WS_RE.sub(" ", kept).strip()


@register
class StopwordRemoval(Compressor):
    """Remove common function words. The classic ~30%-reduction preprocessing."""

    name = "stopwords"
    category = "lexical/filtering"

    def compress(self, text: str) -> str:
        out: list[str] = []
        for tok in _WORD_RE.findall(text):
            if tok.isalpha() and tok.lower() in STOPWORDS:
                continue
            out.append(tok)
        return _join(out)


@register
class SimpleStem(Compressor):
    """Aggressive suffix-stripping stemmer.

    Included to demonstrate the counterintuitive BPE result: normalizing word
    forms ("studies"->"studi", "happily"->"happili") produces strings the
    tokenizer's merges don't cover, so token count often *rises* even though
    character count falls. This is the cautionary tale of the lab.
    """

    name = "stem"
    category = "lexical/normalization"

    _SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ation",
                 "ments", " ness", "ing", "edly", "ies", "ying", "ment", "ness",
                 "ful", "est", "ous", "ive", "ize", "ised", "ed", "ly", "es", "s")

    def _stem(self, word: str) -> str:
        lw = word.lower()
        for suf in self._SUFFIXES:
            suf = suf.strip()
            if len(lw) > len(suf) + 2 and lw.endswith(suf):
                base = lw[: -len(suf)]
                if suf in ("ies", "ied"):
                    base += "i"
                return base
        return lw

    def compress(self, text: str) -> str:
        out = [self._stem(t) if t.isalpha() else t for t in _WORD_RE.findall(text)]
        return _join(out)


def _join(tokens: list[str]) -> str:
    """Re-join tokens, attaching trailing punctuation without a leading space."""
    parts: list[str] = []
    for t in tokens:
        if parts and re.match(r"[^\sA-Za-z\d]", t):
            parts.append(t)
        else:
            parts.append(" " + t if parts else t)
    return "".join(parts).strip()
