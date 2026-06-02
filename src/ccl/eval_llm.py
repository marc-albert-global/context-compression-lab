"""Optional downstream-accuracy evaluation with Claude.

The offline metrics measure *surface* preservation. This measures whether the
compressed text still lets a model answer questions about it, the real test of
"did we keep the context". For each item we give Claude the compressed passage
plus the question and check whether the gold answer is recovered (token-level
F1 ≥ 0.5). Requires `pip install '.[llm]'` and ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import string

from .compressors.base import Compressor
from .corpus import Item

_ARTICLES = {"a", "an", "the"}


def _normalize(s: str) -> list[str]:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in s.split() if t not in _ARTICLES]


def _f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    gg = list(g)
    for t in p:
        if t in gg:
            common += 1
            gg.remove(t)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def _accuracy_once(client, items: list[Item], compressor: Compressor, threshold: float) -> float:
    correct = 0
    for it in items:
        context = compressor.compress(it.passage)
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=64,
            system=[{
                "type": "text",
                "text": "Answer the question using only the provided context. "
                        "Reply with the shortest exact answer span. If unknown, reply 'unknown'.",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {it.question}"}],
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
        if _f1(answer, it.answer) >= threshold:
            correct += 1
    return correct / len(items) if items else 0.0


def task_accuracy(
    items: list[Item], compressor: Compressor, *, threshold: float = 0.5, runs: int = 1
) -> dict:
    """Downstream QA accuracy from the compressed passage, averaged over `runs`.

    LLM answers vary call to call, so a single pass is not trustworthy. Running
    several passes and reporting the mean plus the spread (min/max) shows whether
    a method's accuracy is stable, which is the difference between a number you
    can quote to a stakeholder and one you can't. Returns
    {mean, min, max, runs}, all as fractions in [0, 1].
    """
    import anthropic

    client = anthropic.Anthropic()
    scores = [_accuracy_once(client, items, compressor, threshold) for _ in range(max(1, runs))]
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "runs": len(scores),
    }
