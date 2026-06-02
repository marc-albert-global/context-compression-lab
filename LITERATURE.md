# Context Compression: A Field Survey

A working map of how the field minimizes tokens while trying to preserve
context. Organized by the taxonomy from the NAACL 2025 survey *Prompt
Compression for Large Language Models: A Survey* ([arXiv 2410.12388](https://arxiv.org/abs/2410.12388)).
This is the reference layer of the lab, the **Implemented?** column says which
methods this repo runs directly versus which it surveys (the trained
"soft-prompt" methods need model training and are out of scope for a
dependency-light benchmark).

The two big families: **hard-prompt** methods edit the surface text (drop or
rewrite tokens); **soft-prompt** methods compress context into learned
continuous vectors or special tokens.

---

## 1. Hard-prompt methods: filtering (drop low-value tokens)

| Method | Methodology | Implemented here |
|---|---|---|
| **SelectiveContext** ([Li et al. 2023](https://arxiv.org/abs/2310.06201)) | Score lexical units by **self-information** from a base LM; drop the most predictable (lowest-information) ones. | ✅ approximated (`self-information`, unigram surprisal) |
| **LLMLingua** ([Jiang et al. 2023](https://arxiv.org/abs/2310.05736)) | A small LM (GPT-2/LLaMA-7B) computes per-token **perplexity**; a coarse-to-fine budget controller drops redundant tokens. Up to 20x compression. | 📚 surveyed |
| **LongLLMLingua** ([Jiang et al. 2023](https://arxiv.org/abs/2310.06839)) | Extends LLMLingua for long context with **query-aware** compression, document reordering, and subsequence recovery. | 📚 surveyed |
| **LLMLingua-2** ([Pan et al. 2024](https://arxiv.org/abs/2403.12968)) | Reframes compression as **token classification**; a BERT-level encoder trained on GPT-4-distilled data decides keep/drop. Task-agnostic, 3-6x faster. | 🔌 optional (`[llmlingua]` extra) |
| **AdaComp** | Dynamically selects how many documents/tokens to keep based on query complexity and retrieval quality. | 📚 surveyed |
| **PCRL / TACO-RL** | **Reinforcement learning** to pick which tokens to keep (model-agnostic / task-specific reward). | 📚 surveyed |
| **CPC** | Ranks sentence relevance with context-aware embeddings, keeps the top set. | 📚 surveyed (cf. `tfidf-extractive`) |
| **TCRA-LLM** | Embedding-based summarization + semantic compression. | 📚 surveyed |
| _classic extractive summarization_ | TF-IDF / **TextRank** sentence selection, the pre-LLM workhorse, still a strong baseline. | ✅ (`tfidf-extractive`, `textrank-extractive`) |

## 2. Hard-prompt methods: paraphrasing (rewrite shorter)

| Method | Methodology | Implemented here |
|---|---|---|
| **Nano-Capsulator** | A fine-tuned LM (Vicuna-7B) **paraphrases** the prompt into a shorter natural-language form with a semantic-preservation loss. | 🔌 LLM analog (`llm-abstractive`, `[llm]` extra) |
| **CompAct** | Coarse-to-fine compression driven by perplexity + distilled token scores. | 📚 surveyed |
| **FAVICOMP** | Faithfulness-aware compression for RAG. | 📚 surveyed |

## 3. Soft-prompt methods (compress into learned vectors / tokens)

These require training a model; they are surveyed, not implemented.

| Method | Methodology |
|---|---|
| **GIST** ([Mu et al. 2023](https://arxiv.org/abs/2304.08467)) | Modify attention masks so the prompt compresses into a few trainable **gist tokens** that can be cached and reused; up to 26x compression, generalizes to unseen prompts. |
| **AutoCompressor** | Recursively compress sub-prompts into summary vectors, extending usable context to tens of thousands of tokens. |
| **ICAE** | In-Context Autoencoder: an encoder compresses long context into special tokens, a **frozen** LLM decodes; 4-16x. |
| **500xCompressor** ([Li et al. 2024](https://arxiv.org/abs/2408.03094)) | Compress context into as little as **one token** by feeding the decoder K-V values; 6-480x. |
| **xRAG** ([Cheng et al. 2024](https://arxiv.org/abs/2405.13792)) | Project a document embedding into a **single token** for retrieval-augmented generation via a trained adapter. |
| **COCOM / LLoCO / UniICL / QGC / CC** | Variants compressing documents/demonstrations into context-embedding groups, LoRA params, or projector outputs. |

---

## 4. The non-obvious findings this lab is built around

**Characters saved ≠ tokens saved.** Stemming and lemmatization shrink the
character count but produce word forms outside the BPE merge table
("studies"→"studi", "happily"→"happili"), which fragment into *more* subword
tokens. Empirically, naive stemming can yield **negative** token reduction.
Only stopword removal among classic preprocessing reliably reduces tokens
(~30%). The lab reproduces this directly (`reports/figures/token_vs_char.png`).

**Semantic similarity does not predict downstream accuracy.** The evaluation
study *Understanding and Improving Information Preservation in Prompt
Compression* ([arXiv 2503.19114](https://arxiv.org/abs/2503.19114)) shows a high
reconstruction BERTScore can still go with poor task performance (xRAG: 0.66
BERTScore yet 0.297 EM on HotpotQA), that compression drops grounding by 30-50
points, and that xRAG tokens retain only ~13% of entities. Hence this lab
measures **both** cheap fidelity proxies and an **optional task-accuracy** layer.

**Format is a free lever.** For structured data, Markdown is ~15% cheaper than
JSON, TSV ~50% cheaper, and column-oriented formats (e.g. TOON) 30-60% cheaper
on uniform arrays, before any semantic compression at all.

---

## 5. How this lab evaluates

Mirroring the three-axis framework of arXiv 2503.19114:

1. **Efficiency**: token reduction % (real BPE tokenizer).
2. **Information preservation**: entity retention, ROUGE-L, content-word Jaccard, and optional embedding cosine.
3. **Downstream accuracy**: optional: an LLM answers questions from the compressed vs. full passage (the only test that catches the similarity-vs-accuracy gap).

## Key references

- Li et al., *Prompt Compression for LLMs: A Survey*, NAACL 2025, [2410.12388](https://arxiv.org/abs/2410.12388)
- *Understanding and Improving Information Preservation in Prompt Compression*, EMNLP 2025 Findings, [2503.19114](https://arxiv.org/abs/2503.19114)
- Jiang et al., *LLMLingua*, [2310.05736](https://arxiv.org/abs/2310.05736); *LongLLMLingua*, [2310.06839](https://arxiv.org/abs/2310.06839)
- Pan et al., *LLMLingua-2*, [2403.12968](https://arxiv.org/abs/2403.12968)
- Li et al., *Selective Context*, [2310.06201](https://arxiv.org/abs/2310.06201)
- Mu et al., *Gist Tokens*, [2304.08467](https://arxiv.org/abs/2304.08467)
- Cheng et al., *xRAG*, [2405.13792](https://arxiv.org/abs/2405.13792)
