# Benchmark results

Tokenizer: `tiktoken/cl100k_base`. Corpus: 30 passages.

| Method | Category | Token ↓ % | Preservation % | Entities % | ROUGE-L | Jaccard % |
|---|---|--:|--:|--:|--:|--:|
| `textrank-extractive` | extractive/filtering | 34.9 | 71.4 | 66.6 | 76.2 | 65.3 |
| `tfidf-extractive` | extractive/filtering | 34.7 | 70.7 | 63.5 | 77.9 | 71.8 |
| `self-information` | filtering/information-theoretic | 34.3 | 71.7 | 71.9 | 71.4 | 87.5 |
| `stopwords` | lexical/filtering | 29.1 | 73.8 | 72.6 | 75.0 | 100.0 |
| `punctuation` | lexical/normalization | 8.3 | 95.3 | 93.3 | 97.3 | 93.5 |
| `identity` | baseline | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `dedupe` | format/filtering | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `llm-abstractive` | paraphrasing/LLM (external) | _skipped (needs `llm`)_ | | | | |
| `llmlingua2` | filtering/learned (external) | _skipped (needs `llmlingua`)_ | | | | |
| `whitespace` | lexical/normalization | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| `bullets` | format/rewrite | -1.6 | 97.8 | 95.9 | 99.7 | 99.5 |
| `stem` | lexical/normalization | -6.7 | 46.7 | 13.5 | 79.8 | 50.6 |
