"""Core tests, run offline on the committed corpus."""

import pytest

from ccl import benchmark, corpus, count_tokens, metrics
from ccl.compressors import available, get


def test_corpus_loads():
    items = corpus.load()
    assert len(items) >= 20
    assert all(it.passage and it.question and it.answer for it in items)


def test_registry_has_core_methods():
    names = {c.name for c in available()}
    assert {"identity", "stopwords", "stem", "tfidf-extractive", "self-information", "bullets"} <= names


def test_identity_is_lossless():
    text = "The Eiffel Tower is located in Paris, France."
    assert get("identity").compress(text) == text
    s = metrics.score(text, text)
    assert s.token_reduction == 0.0
    assert s.preservation > 99.0


def test_stopword_removal_reduces_tokens():
    text = "The cat is on the mat and the dog is in the house."
    out = get("stopwords").compress(text)
    assert count_tokens(out) < count_tokens(text)


def test_metrics_bounds():
    orig = "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
    comp = get("stopwords").compress(orig)
    s = metrics.score(orig, comp)
    assert 0 <= s.token_reduction <= 100
    assert 0 <= s.preservation <= 100
    # Entities (Marie, Curie, Nobel, Physics, 1903, ...) should largely survive stopword removal.
    assert s.entity_retention > 80


def test_token_reduction_pct_direction():
    assert metrics.token_reduction_pct("a b c d e f", "a b c") > 0


def test_benchmark_runs_and_ranks():
    results = benchmark.run()
    assert len(results) >= 6
    avail = [r for r in results if r.available]
    # Token reduction can go negative (see test_stemming_backfires); preservation is bounded.
    assert all(-50 <= r.token_reduction <= 100 for r in avail)
    assert all(0 <= r.preservation <= 100 for r in avail)
    # Identity baseline preserves ~everything and reduces nothing.
    identity = next(r for r in results if r.method == "identity")
    assert identity.token_reduction == 0.0
    assert identity.preservation > 99.0


def test_stemming_backfires_under_bpe():
    """The headline finding: aggressive stemming does NOT save tokens (often costs
    them) because it creates word forms the BPE merges don't cover."""
    results = {r.method: r for r in benchmark.run()}
    stem = results["stem"]
    extractive = results["tfidf-extractive"]
    # Stemming reduces far fewer tokens than a real extractive method...
    assert stem.token_reduction < extractive.token_reduction
    # ...and is near-zero or negative, while it badly hurts entity retention.
    assert stem.token_reduction < 5.0
    assert stem.entity_retention < 50.0


def test_external_methods_register_but_skip_when_unavailable():
    results = {r.method: r for r in benchmark.run()}
    # llm-abstractive needs a key; in CI it should be present but marked unavailable.
    assert "llm-abstractive" in results
    if not results["llm-abstractive"].available:
        assert results["llm-abstractive"].requires_extra == "llm"
