---
title: "Step 8: BM25: search and compare"
series: "Simple RAG"
step: 8
slug: "step-08-bm25-search-and-compare"
date: "2026-10-21"
tag: "RAG"
excerpt: "Give BM25 a search with zero new code, build a tool that runs every method side by side, and catch TF-IDF being fooled by spam."
minutes: 8
---

# Step 8: BM25: search and compare

**You'll build:** `search()` in `bm25.py`, and `src/ragi/retrieval/compare.py`.
**You'll learn:** why the same code behaves differently in different files, modules as values, and f-string number formatting.

---

## 1. Theory

All three methods follow the same recipe: score every document → drop the zeros → sort → keep the top k. Only `score` differs. So BM25's `search` is the **exact same code** as TF-IDF's, and a small tool can run every method on the same question so you can see where each one wins and fails.

---

## Before vs after (measured)

| Spam test: `"python python … python"` added, query `"python snake"` | Before (TF-IDF) | After (BM25) |
|---|---|---|
| 1st | `spam.txt` 0.511 ❌ | `python_snake.txt` 2.245 ✅ |
| 2nd | `python_snake.txt` 0.094 | `spam.txt` 1.170 |

---

## 2. Code example

### 2.1 Why copy-paste works here
Each file has its own names. Inside `bm25.py`, the name `score` means **bm25's** `score`; inside `tfidf.py`, it means TF-IDF's. So the same `search` code, pasted into `bm25.py`, automatically uses BM25. Python looks up names in the file the code lives in.

### 2.2 `compare.py`: given
It's just a printing tool, so all of it is given:
```python
from ragi.loaders import load_folder
from ragi.retrieval import keyword, tfidf, bm25      # import whole files (modules), not single functions

docs = load_folder()
while True:
    query = input("compare> ")
    for name, method in [("keyword", keyword), ("tfidf", tfidf), ("bm25", bm25)]:
        print(f"--- {name}")
        for s, doc in method.search(query, docs):    # each module's own search
            print(f"  {s:.3f}  {doc.source}")        # :.3f = show 3 decimals
```

| Piece | Meaning |
|---|---|
| `from ragi.retrieval import keyword, tfidf, bm25` | imports the **files themselves**, not single functions |
| `[("keyword", keyword), ...]` | a **module is a value** too, so you can put modules in a list and loop over them |
| `method.search(...)` | calls *that* module's `search` |
| `f"{s:.3f}"` | formats the number with 3 decimals: `2.2451` → `2.245` |

---

## 3. Goal

1. Copy `search` **and** the `if __name__ == "__main__":` box from `tfidf.py` into `bm25.py`. Add `from ragi.loaders import Document` at the top. **Zero lines change.**
2. Create `src/ragi/retrieval/compare.py` with the code above.

### ✅ Checkpoint 1: BM25 alone
```bash
uv run python -m ragi.retrieval.bm25
```
Type `python snake`: `python_snake.txt` should come first.

### ✅ Checkpoint 2: the spam experiment
1. Create `data/spam.txt` containing only:
   ```
   python python python python python python python python
   ```
2. Run the compare tool and type **`python snake`**:
   ```bash
   uv run python -m ragi.retrieval.compare
   ```

| Method | 1st | 2nd |
|---|---|---|
| keyword | python_snake.txt (2) ✅ | python_language / spam (1) |
| **tfidf** | **spam.txt (0.511)** ❌ fooled | python_snake.txt (0.094) |
| **bm25** | **python_snake.txt (2.245)** ✅ | spam.txt (1.170) |

TF-IDF ranks a file of eight repeated words **5× higher** than the real snake document. BM25's saturation caps how much repetition can earn. (Keyword search isn't fooled either: it counts *which* question words appear, not how often.)

3. Try **`how long can a python sleep`**. All three get it wrong in different ways: they rank documents by shared words like "can" and "sleep". None of them knows the question is about a *snake* sleeping. That's the meaning problem, and semantic search (Step 9) solves it.

4. **Delete `data/spam.txt` afterwards.** Some tests use the real `data/` folder and fail while it's there.

---

## 4. Test

**This test checks:** BM25 isn't fooled by repetition. **How:** a doc that's only "python" × 8 must lose to a real doc that mentions both "python" and "snake" once.
```python
def test_search_is_not_fooled_by_repetition():
    docs = [
        Document(text="python python python python python python python python", source="spam.txt"),
        Document(text="the python is a large snake", source="snake.txt"),
        Document(text="coffee beans", source="coffee.txt"),
        Document(text="felines purr", source="cat.txt"),
    ]
    results = search("python snake", docs)
    assert results[0][1].source == "snake.txt"
```

**This test checks:** zero scores are dropped, and you never get more than k results. **How:** only one doc contains "snake"; four docs match the long query, but `k=2`.
```python
def test_search_drops_zero_scores_and_limits_k():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="python snake", source="snake.txt"),
        Document(text="python tea", source="tea.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    assert len(search("snake", docs, k=3)) == 1
    assert len(search("python snake code tea", docs, k=2)) == 2
```

Imports: `from ragi.loaders import Document` and `from ragi.retrieval.bm25 import search`.

```bash
uv run pytest -v
```

---

## 5. Make it a module

You now have **three interchangeable search methods** with the same `search(query, docs, k)` shape. In Step 10 you add a fourth, semantic search, to `compare.py` as one more item in that list. In Step 12, the RAG pipeline picks one of them.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/bm25.py</code> and <code>src/ragi/retrieval/compare.py</code></strong></summary>

```python
# src/ragi/retrieval/bm25.py
from ragi.retrieval.keyword import tokenize
from ragi.retrieval.tfidf import idf
from ragi.loaders import Document


def term_score(f: int, doc_len: int, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    # 1 = average length, 2 = twice as long
    length_ratio = doc_len / avg_len
    return f * (k1 + 1) / (f + k1 * (1 - b + b * length_ratio))


def score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float:
    if len(all_doc_words) == 0:
        return 0.0

    query_words = tokenize(query)
    # total words of all docs / number of docs
    avg_len = sum([len(w) for w in all_doc_words]) / len(all_doc_words)
    total = 0.0
    for word in query_words:
        f = doc_words.count(word)
        total = total + idf(word, all_doc_words) * \
            term_score(f, len(doc_words), avg_len)

    return total


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]:
    all_doc_words = [tokenize(d.text) for d in docs]
    results = []
    for doc in docs:
        doc_words = tokenize(doc.text)
        s = score(query, doc_words, all_doc_words)
        if s > 0:
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
from ragi.retrieval import keyword, tfidf, bm25

docs = load_folder()
while True:
    query = input("compare> ")
    for name, method in [("keyword", keyword), ("tfidf", tfidf), ("bm25", bm25)]:
        print(f"--- {name}")
        for s, doc in method.search(query, docs):    # each module's own search
            print(f"  {s:.3f}  {doc.source}")        # :.3f = show 3 decimals
```

</details>

---

**Previous:** [← Step 7: BM25: the formula](step-07-bm25-the-formula.md)

**Next:** [Step 9: Embeddings and similarity →](step-09-embeddings-and-similarity.md)
