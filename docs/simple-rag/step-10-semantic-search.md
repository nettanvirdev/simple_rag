---
title: "Step 10: Semantic search"
series: "Simple RAG"
step: 10
slug: "step-10-semantic-search"
date: "2026-10-21"
tag: "RAG"
excerpt: "Rank documents by meaning instead of shared words, and watch semantic search find the cat document for \"cat\" when every word-based method finds nothing."
minutes: 10
---

# Step 10: Semantic search

**You'll build:** `search()` in `src/ragi/retrieval/semantic.py`, plus semantic search as the 4th method in `compare.py`.
**You'll learn:** embedding many texts at once, reading a similarity table, `zip`, and why semantic search never returns "nothing".

---

## 1. Theory

Semantic search embeds every document and the question, then ranks the documents by cosine similarity. It's the same recipe as Steps 4, 6 and 8 (score everything → sort → top k); only the score changes. One big difference: there's no "zero". Every document has *some* similarity, so semantic search **always** returns k results, even for nonsense.

More theory: [concepts.md, Semantic search](../concepts.md#6-semantic-search).

---

## Before vs after (measured)

| Question | Before (keyword / TF-IDF / BM25) | After (semantic search) |
|---|---|---|
| `cat` | nothing found | `felines.txt` 0.459 |
| `what helps me stay awake` | nothing found | `coffee.txt` 0.356 |
| `a reptile that squeezes its prey` | weak word matches | `python_snake.txt` 0.398 |

---

## 2. Code example

### 2.1 Three library lines (given)
```python
doc_vectors = model.encode([d.text for d in docs])          # ALL docs at once → a table of shape (4, 384): one row per doc
query_vector = model.encode(query)                          # the question → (384,)
scores = util.cos_sim(query_vector, doc_vectors)[0].tolist()
```

The last line, piece by piece:

| Piece | Result |
|---|---|
| `cos_sim(query, docs)` | a 1×4 table: `[[0.46, 0.17, 0.16, 0.13]]` (1 row for the question, 4 columns for the docs) |
| `[0]` | its first (and only) row: `[0.46, 0.17, 0.16, 0.13]` |
| `.tolist()` | converts that row from a tensor to a normal Python list |

So `scores[0]` is the score for `docs[0]`, `scores[1]` is the score for `docs[1]`, and so on.

Encoding all documents in **one** call is much faster than calling `encode` once per document.

### 2.2 New idea: `zip`
`zip` walks through two lists side by side:
```python
for s, doc in zip(scores, docs):    # (0.46, coffee doc), (0.17, felines doc), ...
    print(s, doc.source)
```

---

## 3. Goal

Add `search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]` to `semantic.py`:
1. The 3 lines above.
2. `results = []`. Loop with `zip(scores, docs)` and append `(s, doc)` for **every** document. Don't use `if s > 0`: similarity has no "no match".
3. Sort by score, highest first, and return the top k, the same as your other searches.
4. Copy the `if __name__ == "__main__":` search box from `keyword.py` to the bottom.

Then, in `compare.py`, add `semantic` to the import line and `("semantic", semantic)` to the list of methods.

### ✅ Checkpoint
```bash
uv run python -m ragi.retrieval.compare
```
It starts a few seconds slower now, because it loads the model.

| Type | Semantic ranks first | Keyword / TF-IDF / BM25 |
|---|---|---|
| `cat` | **felines.txt** (0.459) | nothing; the word "cat" isn't in any file |
| `what helps me stay awake` | **coffee.txt** (0.356) | nothing |
| `a reptile that squeezes its prey` | **python_snake.txt** (0.398) | weak: keyword ties on "a", TF-IDF and BM25 only match "prey" |
| `asdfgh` | still 3 results (low scores) | nothing |

The last row matters for RAG: semantic search **always** answers, even when nothing is relevant.

---

## 4. Test

**This test checks:** semantic search finds meaning without shared words. **How:** "cat" must rank `felines.txt` first, a file that never contains the word "cat", so only meaning can put it on top.

**This test checks:** semantic search always returns k results. **How:** a nonsense query like `"asdfgh"` still returns exactly 3, where keyword search would return nothing.

```python
# tests/test_semantic.py (added to the Step 9 tests)
from ragi.loaders import Document, load_folder
from ragi.retrieval.semantic import search


def test_search_finds_meaning_without_shared_words():
    results = search("cat", load_folder())
    assert results[0][1].source == "felines.txt"


def test_search_always_returns_k_results():
    docs = [Document(text=t, source=f"{i}.txt") for i, t in enumerate(["coffee", "tea", "snake", "kitten"])]
    assert len(search("asdfgh", docs, k=3)) == 3
```

`enumerate` gives each item a number: `(0, "coffee")`, `(1, "tea")`, ... so every fake doc gets its own file name.

```bash
uv run pytest -v
```

---

## 5. Make it a module

All 4 search methods now share the **same shape**, `search(query, docs, k)` → `[(score, Document), ...]`, so any of them can be plugged into the RAG pipeline. Step 12 uses semantic search.

Notice that `search` **re-embeds every document on every question**. That's fine for 4 tiny files and impossible for 10,000 PDFs. Storing those vectors once and reusing them is exactly what a **vector database** (like ChromaDB) is for.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/semantic.py</code> and <code>src/ragi/retrieval/compare.py</code></strong></summary>

```python
# src/ragi/retrieval/semantic.py
from sentence_transformers import SentenceTransformer, util
from ragi import config
from ragi.loaders import Document

model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(text: str):
    return model.encode(text)


def similarity(a: str, b: str) -> float:
    similarity = util.cos_sim(embed(a), embed(b)).item()
    return similarity


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]:
    # ALL docs at once → table of shape (4, 384), one row per doc
    doc_vectors = model.encode([d.text for d in docs])
    # the question → (384,)
    query_vector = model.encode(query)
    scores = util.cos_sim(query_vector, doc_vectors)[0].tolist()
    results = []
    # re-embeds every doc on every search; a vector DB stores them once
    for s, doc in zip(scores, docs):
        results.append((s, doc))

    return sorted(results, key=lambda s: s[0], reverse=True)[:k]


# only runs when you start this file directly, not when another file imports it
if __name__ == "__main__":
    from ragi.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
```

```python
# src/ragi/retrieval/compare.py
from ragi.loaders import load_folder
# import whole files (modules), not single functions
from ragi.retrieval import keyword, tfidf, bm25, semantic

docs = load_folder()
while True:
    query = input("compare> ")
    for name, method in [("keyword", keyword), ("tfidf", tfidf), ("bm25", bm25), ("semantic", semantic)]:
        print(f"--- {name}")
        for s, doc in method.search(query, docs):    # each module's own search
            print(f"  {s:.3f}  {doc.source}")        # :.3f = show 3 decimals
```

</details>

---

**Previous:** [← Step 9: Embeddings and similarity](step-09-embeddings-and-similarity.md)

**Next:** [Step 11: Talking to an LLM →](step-11-talking-to-an-llm.md)
