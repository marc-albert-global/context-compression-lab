# context-compression-lab

[![CI](https://github.com/marc-albert-global/context-compression-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/marc-albert-global/context-compression-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Cut the token bill of a GenAI application without quietly breaking it.** This
is a benchmark lab that runs many context-compression methods over a corpus and
measures, for each one, how much it reduces tokens against how much context it
preserves, then turns that into a dollar projection at your volume.

It exists because "make the LLM cheaper" is a real production ask, and the only
honest way to answer it is with numbers: which technique saves how much, where
quality starts to break, and what that is worth per year.

> Scope: a reproducible study on public text (SQuAD passages), not a production
> system. The methods and the measurement are the deliverable.

---

## Impact at a glance

| | |
|---|---|
| Token reduction (best safe method) | **~35%** at ~71% context preservation |
| Cheap safe win | **punctuation strip: 8% fewer tokens at 95% preservation** |
| Projected saving | **~$6,400/yr** at 10M input tokens/day (Opus $5/1M), from a 35% cut |
| Cautionary result | aggressive stemming **costs** 6.7% more tokens (BPE fragmentation) |
| Methods benchmarked | 10, pluggable; new method = one class + one decorator |

Every number above is produced by a committed script on committed data. Run
`ccl bench` and `ccl cost` to reproduce them.

---

## Problem and context

Teams ship a GenAI feature, the token bill scales with usage, and someone asks
"can we make this cheaper?" The naive answer (truncate the context) silently
degrades quality. The real question has three axes at once: **how many tokens
you save, whether quality survives, and what the saving is worth.** Most "we cut
costs X%" claims report only the first. This lab reports all three, and is built
to expose the case the literature warns about: semantic similarity staying high
while task accuracy quietly drops.

## Approach

A compressor takes text and returns shorter text. Each one is scored on:

- **Efficiency**: token reduction %, measured with a real BPE tokenizer (`tiktoken`), so byte-pair effects are genuine.
- **Preservation**: entity retention and ROUGE-L against the original (the headline "preservation %" is their mean), plus optional embedding cosine.
- **Downstream accuracy** (optional, LLM): does a model still answer questions correctly from the compressed text? This is the only metric that catches the similarity-vs-accuracy gap, so it is kept separate and honest rather than folded into a single score.

Methods span the families in the field survey ([LITERATURE.md](LITERATURE.md)):
lexical preprocessing (stopwords, punctuation, stemming), statistical filtering
(TF-IDF and TextRank extraction, self-information pruning à la SelectiveContext),
and format rewriting, with optional hooks for Microsoft's LLMLingua-2 and an LLM
rewriter.

## Results

Corpus: 30 SQuAD passages. Tokenizer: `tiktoken/cl100k_base`.

| Method | Token ↓ % | Preservation % | Entities % |
|---|--:|--:|--:|
| `textrank-extractive` | **34.9** | 71.4 | 66.6 |
| `tfidf-extractive` | 34.7 | 70.7 | 63.5 |
| `self-information` | 34.3 | 71.7 | 71.9 |
| `stopwords` | 29.1 | 73.8 | 72.6 |
| `punctuation` | 8.3 | **95.3** | 93.3 |
| `stem` | **-6.7** | 46.7 | 13.5 |

![Pareto frontier](reports/figures/pareto.png)

**Stemming backfires.** It cuts characters but *increases* tokens, because
normalizing word forms ("studies" to "studi") produces strings the BPE merges
don't cover. Characters saved is not tokens saved:

![Characters vs tokens](reports/figures/token_vs_char.png)

## From demo to deployment

How this maps to a real cost-reduction engagement:

- **Pick the operating point from the Pareto frontier, not a vibe.** For a retrieval/RAG context where entities must survive, `self-information` (34% off, 72% entities) is defensible; for a tolerant summarization step, extraction goes further; where fidelity is paramount, `punctuation` is a free 8%.
- **Decision rule, not a default.** The lab outputs the trade-off so an operator can set a fidelity floor (e.g. "preservation must stay ≥ 70%") and take the cheapest method above it.
- **Cost model at your scale** (`ccl cost --volume N --price P`):

  | Volume (input tokens/day) | 35% reduction saves |
  |---|--:|
  | 1,000,000 | ~$640 / yr |
  | 10,000,000 | ~$6,400 / yr |
  | 100,000,000 | ~$64,000 / yr |

  (Projections at Opus $5/1M input; swap `--price` for any model.)
- **What to monitor in production**: token reduction realized vs. expected, and the downstream task metric, so a quality regression is caught rather than discovered on the bill.

## Methodology

- **Baseline** is explicit: the `identity` compressor (0% reduction, 100% preservation) anchors the scale.
- **"Preservation"** is the mean of entity retention and ROUGE-L F1 against the original. It is a *proxy*; the offline metrics are deterministic, so they have no run-to-run variance.
- **Downstream accuracy is the real test** and it is not deterministic: LLM answers vary per call. `ccl bench --llm-eval --runs 5` averages several passes and reports the spread, because a single pass is not quotable.
- **We report a method that fails.** `stem` is kept in the table precisely because a benchmark where everything wins reads as marketing.

### The "did quality hold?" proof (ready to run)

The offline metrics show context is *preserved*; proving *task accuracy* held
needs the LLM layer. It is built and ready:

```bash
pip install -e ".[llm]" && export ANTHROPIC_API_KEY=sk-ant-...
ccl bench --llm-eval --runs 5     # answers SQuAD questions from compressed vs full context
```

This populates a `task_accuracy` column so you can confirm the headline claim:
tokens down, accuracy flat.

## Quickstart

```bash
git clone https://github.com/marc-albert-global/context-compression-lab.git
cd context-compression-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ccl methods                                  # list compressors
ccl compress -m stopwords "The cat sat on the mat in the warm afternoon sun."
ccl bench                                    # full benchmark -> reports/ + figures
ccl cost --volume 10000000                   # project savings at your scale
pytest -q                                    # 11 tests, offline
```

## Add a method (the continuous-research part)

```python
from ccl.compressors.base import Compressor, register

@register
class DropVowels(Compressor):
    name = "drop-vowels"
    category = "lexical/experimental"
    def compress(self, text: str) -> str:
        return "".join(c for c in text if c.lower() not in "aeiou")
```

It appears in `ccl methods`, the benchmark, and the cost model automatically.

## Limitations

- Preservation proxies are not task accuracy; the LLM layer (above) is what closes that gap, and it has not been run in the committed results yet.
- Corpus is 30 English passages; results will shift on other domains and languages.
- Trained "soft-prompt" compressors (LLMLingua-2, ICAE, xRAG) are surveyed, not reimplemented; LLMLingua-2 is wired as an optional comparator.

## Roadmap

- Run and publish the multi-run LLM task-accuracy numbers.
- Add the LLMLingua-2 and embedding-cosine columns to the default report.
- Expand the corpus (multi-domain, multi-language) and add a length-stratified view.
- A `--fidelity-floor` flag that picks the cheapest method above a quality bar.

## Data

30 passage+QA items from SQuAD v1.1 dev (CC BY-SA 4.0), committed for offline
reproducibility. Provenance in [`data/README.md`](data/README.md). Field survey
of the literature in [`LITERATURE.md`](LITERATURE.md).

## License

MIT © 2026 Marc Albert. Bundled data © its original authors under CC BY-SA 4.0.
