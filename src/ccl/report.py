"""Render benchmark results: a markdown/CSV table and figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import corpus
from .benchmark import Result
from .compressors import get
from .tokenization import count_tokens, tokenizer_name

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"
FIG_DIR = REPORTS / "figures"

_COLS = ["method", "category", "token_reduction", "preservation",
         "entity_retention", "rouge_l", "content_jaccard"]


def markdown_table(results: list[Result]) -> str:
    rows = ["| Method | Category | Token ↓ % | Preservation % | Entities % | ROUGE-L | Jaccard % |",
            "|---|---|--:|--:|--:|--:|--:|"]
    for r in sorted(results, key=lambda x: x.token_reduction, reverse=True):
        if not r.available:
            rows.append(f"| `{r.method}` | {r.category} | _skipped (needs `{r.requires_extra}`)_ | | | | |")
            continue
        rows.append(
            f"| `{r.method}` | {r.category} | {r.token_reduction:.1f} | "
            f"{r.preservation:.1f} | {r.entity_retention:.1f} | {r.rouge_l:.1f} | "
            f"{r.content_jaccard:.1f} |"
        )
    return "\n".join(rows)


def write_all(results: list[Result]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(REPORTS / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "category", "available", "token_reduction", "preservation",
                    "entity_retention", "rouge_l", "content_jaccard", "embedding_cosine",
                    "task_accuracy"])
        for r in results:
            w.writerow([r.method, r.category, r.available, r.token_reduction, r.preservation,
                        r.entity_retention, r.rouge_l, r.content_jaccard, r.embedding_cosine,
                        r.task_accuracy])

    # Markdown
    (REPORTS / "results.md").write_text(
        f"# Benchmark results\n\nTokenizer: `{tokenizer_name()}`. "
        f"Corpus: {len(corpus.load())} passages.\n\n" + markdown_table(results) + "\n",
        encoding="utf-8",
    )

    _plot_pareto(results)
    _plot_token_vs_char()


def _plot_pareto(results: list[Result]) -> None:
    pts = [r for r in results if r.available and r.method != "identity"]
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [r.token_reduction for r in pts]
    ys = [r.preservation for r in pts]
    ax.scatter(xs, ys, s=80, color="#2563eb", zorder=3)
    for r in pts:
        ax.annotate(r.method, (r.token_reduction, r.preservation),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    # Pareto frontier: points not dominated on both axes.
    frontier: list[Result] = []
    for r in sorted(pts, key=lambda x: x.token_reduction, reverse=True):
        if all(r.preservation >= f.preservation for f in frontier):
            frontier.append(r)
    frontier.sort(key=lambda x: x.token_reduction)
    if len(frontier) > 1:
        ax.plot([f.token_reduction for f in frontier], [f.preservation for f in frontier],
                color="#dc2626", lw=1.5, ls="--", zorder=2, label="Pareto frontier")
        ax.legend()
    ax.set_xlabel("Token reduction % (higher = cheaper)")
    ax.set_ylabel("Context preservation % (higher = more faithful)")
    ax.set_title("Compression methods: efficiency vs. fidelity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto.png", dpi=130)
    plt.close(fig)


def _plot_token_vs_char() -> None:
    """Reproduce the BPE finding: stemming cuts characters but not tokens."""
    items = corpus.load()
    methods = ["whitespace", "punctuation", "stopwords", "stem", "self-information"]
    char_red, tok_red = [], []
    for m in methods:
        comp = get(m)
        co = ct = oo = ot = 0
        for it in items:
            c = comp.compress(it.passage)
            oo += len(it.passage)
            co += len(c)
            ot += count_tokens(it.passage)
            ct += count_tokens(c)
        char_red.append((1 - co / oo) * 100)
        tok_red.append((1 - ct / ot) * 100)

    import numpy as np

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, char_red, 0.4, label="character reduction %", color="#93c5fd")
    ax.bar(x + 0.2, tok_red, 0.4, label="token reduction %", color="#2563eb")
    ax.axhline(0, color="#374151", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("reduction %")
    ax.set_title(
        "Characters saved is not tokens saved (BPE effect)\n"
        "stemming cuts characters but barely tokens"
    )
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_vs_char.png", dpi=130)
    plt.close(fig)
