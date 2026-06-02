"""Extractive and information-theoretic compressors (hard-prompt, statistical).

These are simplified, dependency-free cousins of methods in the literature:
- TF-IDF / TextRank sentence selection are classic extractive summarization.
- SelfInfoPrune approximates SelectiveContext (Li et al., 2023), which keeps
  high-self-information tokens. We estimate self-information from the passage's
  own unigram distribution rather than a separate language model, so it runs
  offline; the idea (drop the most predictable, lowest-information tokens) is
  the same.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import Compressor, register
from .lexical import STOPWORDS, _join

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _content_terms(sentence: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+", sentence) if w.lower() not in STOPWORDS]


@register
class TfidfSelector(Compressor):
    """Keep the top-scoring half of sentences by summed TF-IDF of content terms."""

    name = "tfidf-extractive"
    category = "extractive/filtering"
    ratio = 0.5

    def compress(self, text: str) -> str:
        sents = _split_sentences(text)
        if len(sents) <= 2:
            return text
        df: Counter[str] = Counter()
        per_sent_terms = []
        for s in sents:
            terms = _content_terms(s)
            per_sent_terms.append(terms)
            df.update(set(terms))
        n = len(sents)
        scores = []
        for terms in per_sent_terms:
            tf = Counter(terms)
            score = sum((c / max(len(terms), 1)) * math.log(n / (1 + df[t])) for t, c in tf.items())
            scores.append(score)
        keep = max(1, round(n * self.ratio))
        top_idx = sorted(sorted(range(n), key=lambda i: scores[i], reverse=True)[:keep])
        return " ".join(sents[i] for i in top_idx)


@register
class TextRankSelector(Compressor):
    """Keep the top half of sentences by TextRank centrality over term overlap."""

    name = "textrank-extractive"
    category = "extractive/filtering"
    ratio = 0.5

    def compress(self, text: str) -> str:
        sents = _split_sentences(text)
        n = len(sents)
        if n <= 2:
            return text
        term_sets = [set(_content_terms(s)) for s in sents]
        # Similarity = normalized term overlap.
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = term_sets[i], term_sets[j]
                if a and b:
                    overlap = len(a & b) / (math.log(len(a) + 1) + math.log(len(b) + 1) + 1e-9)
                    sim[i][j] = sim[j][i] = overlap
        scores = [1.0] * n
        d = 0.85
        for _ in range(30):  # power iteration
            new = []
            for i in range(n):
                inbound = sum(sim[i][j] / (sum(sim[j]) or 1e-9) * scores[j] for j in range(n) if j != i)
                new.append((1 - d) + d * inbound)
            scores = new
        keep = max(1, round(n * self.ratio))
        top_idx = sorted(sorted(range(n), key=lambda i: scores[i], reverse=True)[:keep])
        return " ".join(sents[i] for i in top_idx)


@register
class SelfInfoPrune(Compressor):
    """Drop the lowest-self-information tokens (SelectiveContext-style).

    Self-information of a token is estimated as -log p(token) under the
    passage's unigram distribution; the most frequent (most predictable) tokens
    carry the least information and are pruned below a percentile threshold.
    Stopwords and high-information tokens are kept.
    """

    name = "self-information"
    category = "filtering/information-theoretic"
    drop_fraction = 0.35

    def compress(self, text: str) -> str:
        toks = _WORD_RE.findall(text)
        words = [t for t in toks if t.isalpha()]
        if len(words) < 8:
            return text
        total = len(words)
        freq = Counter(w.lower() for w in words)
        info = {w: -math.log(freq[w.lower()] / total) for w in set(words)}
        # Threshold: drop the lowest-information fraction of *word* tokens.
        ordered = sorted(info.values())
        cut = ordered[min(len(ordered) - 1, int(len(ordered) * self.drop_fraction))]
        out = []
        for t in toks:
            if t.isalpha() and info.get(t, 99) < cut:
                continue
            out.append(t)
        return _join(out)
