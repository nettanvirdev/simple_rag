---
title: "Simple RAG"
series: "Simple RAG"
step: 0
slug: "index"
date: "2026-10-21"
tag: "RAG"
excerpt: "Learn RAG by building it: keyword search, TF-IDF, BM25 and semantic search from scratch in Python, then an LLM that answers from your own documents with citations."
minutes: 3
---

# Simple RAG

Build a Retrieval-Augmented Generation system in plain Python, one small file at a time. You'll write four search engines yourself (**keyword**, **TF-IDF**, **BM25**, **semantic**), compare them side by side, and finish with an LLM that answers questions from *your* documents, cites its sources, and says "I don't know" when the answer isn't there.

**You need:** basic Python (variables, functions, loops, input/output). Everything else (dataclasses, regex, lambdas, generators, dictionaries, embeddings, mocking) is explained the first time it appears.

## How every step works
1. **Theory**: 2–3 lines on the idea, plus real numbers.
2. **Before vs after (measured)**: what this step changes, in real numbers.
3. **Code example**: the new or tricky pieces, explained line by line. Never the full solution.
4. **Goal**: you write the code from plain-English steps, then run a **checkpoint** command with the expected output.
5. **Test**: pytest tests that prove it works, with one sentence each on *what* they check and *why* the setup proves it.
6. **Make it a module**: how the next step imports what you just built.
7. **Show solution**: the complete file, collapsed. Open it only if you're stuck.

## The steps
| # | Step | You build | You learn |
|---|---|---|---|
| 1 | [Project setup](step-01-project-setup.md) | `config.py` | uv, `.env`, `pathlib`, first pytest |
| 2 | [Loading text files](step-02-loading-text-files.md) | `loaders.py` | dataclasses, type hints, generators, `tmp_path` |
| 3 | [Keyword search: tokenize and score](step-03-keyword-tokenize-and-score.md) | `retrieval/keyword.py` | regex, raw strings, `in` on strings vs. lists |
| 4 | [Keyword search: ranking](step-04-keyword-search.md) | `search()` | tuples, sorting with `lambda`, slicing, `__main__` |
| 5 | [TF-IDF: the math](step-05-tfidf-the-math.md) | `retrieval/tfidf.py` | term frequency, inverse document frequency, `log` |
| 6 | [TF-IDF: search](step-06-tfidf-search.md) | `score()`, `search()` | Σ tf × idf, breaking keyword ties |
| 7 | [BM25: the formula](step-07-bm25-the-formula.md) | `retrieval/bm25.py` | saturation (k1), length fairness (b) |
| 8 | [BM25: search and compare](step-08-bm25-search-and-compare.md) | `compare.py` | modules as values, the spam test |
| 9 | [Embeddings and similarity](step-09-embeddings-and-similarity.md) | `retrieval/semantic.py` | vectors, cosine similarity, tensors |
| 10 | [Semantic search](step-10-semantic-search.md) | `search()` | `zip`, search by meaning |
| 11 | [Talking to an LLM](step-11-talking-to-an-llm.md) | `llm.py` | chat roles, dictionaries, mocking |
| 12 | [Your first RAG](step-12-your-first-rag.md) | `rag.py` | retrieve → augment → generate |

The theory and math for every idea, plus a one-page cheat sheet, are in **[RAG Concepts](../concepts.md)**. The whole pipeline, with every function and model labelled, is in **[Diagrams](diagrams.md)**.

When you're done, continue with **[Advanced RAG](https://github.com/nettanvirdev/advanced_rag)**: chunking, ChromaDB, hybrid search, reranking, caching, evaluation and FastAPI.

## Practice data
Four tiny files in `data/`, designed to show where each method wins and fails:

| File | Why it's there |
|---|---|
| `python_language.txt` | "python" the programming language |
| `python_snake.txt` | "python" the snake: same word, different meaning |
| `felines.txt` | about cats, but never uses the word "cat" |
| `coffee.txt` | shares filler words ("is", "a") with everything |

**Start here:** [Step 1: Project setup →](step-01-project-setup.md)
