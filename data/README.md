# Data provenance

| | |
|---|---|
| **File** | `passages.jsonl` (30 items) |
| **Source** | [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) development set (Rajpurkar et al., 2016) |
| **Underlying text** | Wikipedia passages |
| **License** | CC BY-SA 4.0 |
| **Sampling** | 30 passages of 400-900 characters, each from a distinct article, one question + reference answer per passage (seed 42). |

Each line: `{id, title, passage, question, answer}`. The passages are the unit
of compression; the question/answer pairs power the optional LLM task-accuracy
evaluation. Committed so the benchmark reproduces offline.

To re-sample from the source:

```bash
curl -s https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -o dev.json
# then select passages by length and distinct title (see project history)
```
