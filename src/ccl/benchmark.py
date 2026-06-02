"""Run every registered compressor over the corpus and aggregate metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import corpus, metrics
from .compressors import available


@dataclass
class Result:
    method: str
    category: str
    available: bool
    token_reduction: float = 0.0
    preservation: float = 0.0
    entity_retention: float = 0.0
    rouge_l: float = 0.0
    content_jaccard: float = 0.0
    embedding_cosine: float | None = None
    task_accuracy: float | None = None  # optional, from LLM eval
    requires_extra: str | None = None


def run(*, embedding: bool = False, methods: list[str] | None = None) -> list[Result]:
    items = corpus.load()
    compressors = [c for c in available() if methods is None or c.name in methods]
    results: list[Result] = []

    for comp in compressors:
        if not comp.is_available():
            results.append(Result(comp.name, comp.category, available=False,
                                   requires_extra=comp.requires_extra))
            continue
        agg = {"token_reduction": 0.0, "entity_retention": 0.0, "rouge_l": 0.0,
               "content_jaccard": 0.0, "preservation": 0.0}
        emb_vals: list[float] = []
        for it in items:
            compressed = comp.compress(it.passage)
            s = metrics.score(it.passage, compressed, embedding=embedding)
            agg["token_reduction"] += s.token_reduction
            agg["entity_retention"] += s.entity_retention
            agg["rouge_l"] += s.rouge_l
            agg["content_jaccard"] += s.content_jaccard
            agg["preservation"] += s.preservation
            if s.embedding_cosine is not None:
                emb_vals.append(s.embedding_cosine)
        n = len(items)
        results.append(Result(
            method=comp.name,
            category=comp.category,
            available=True,
            token_reduction=round(agg["token_reduction"] / n, 1),
            preservation=round(agg["preservation"] / n, 1),
            entity_retention=round(agg["entity_retention"] / n, 1),
            rouge_l=round(agg["rouge_l"] / n, 1),
            content_jaccard=round(agg["content_jaccard"] / n, 1),
            embedding_cosine=round(sum(emb_vals) / len(emb_vals), 1) if emb_vals else None,
            requires_extra=comp.requires_extra,
        ))
    return results


def add_task_accuracy(
    results: list[Result], *, methods: list[str] | None = None, runs: int = 1
) -> list[Result]:
    """Augment results with LLM-based downstream task accuracy (optional)."""
    from . import eval_llm

    items = corpus.load()
    by_name = {r.method: r for r in results}
    from .compressors import available as _avail

    comps = {c.name: c for c in _avail()}
    for name, r in by_name.items():
        if not r.available:
            continue
        if methods is not None and name not in methods:
            continue
        acc = eval_llm.task_accuracy(items, comps[name], runs=runs)
        r.task_accuracy = round(acc["mean"] * 100, 1)
    return results


def to_dicts(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]


if __name__ == "__main__":
    from . import report

    res = run()
    report.write_all(res)
    print(report.markdown_table(res))
