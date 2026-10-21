---
title: "Step 6: TF-IDF: search"
series: "Simple RAG"
step: 6
slug: "step-06-tfidf-search"
date: "2026-10-21"
tag: "RAG"
excerpt: "Turn tf and idf into a real TF-IDF search engine and watch it break the ties that keyword search can't."
minutes: 10
---

# Step 6: TF-IDF: search

**You'll build:** `score()` and `search()` in `src/ragi/retrieval/tfidf.py`.
**You'll learn:** the TF-IDF score (Σ tf × idf), what goes above a loop vs. inside it, and why a `NameError` happens.

---

## 1. Theory

A document's TF-IDF score for a question is the **sum of `tf × idf` for every word in the question**. Rare words add a lot, and common words like "a" add almost nothing. Search then works exactly like keyword search: score every document, drop the zeros, sort, and keep the top k. Only the scoring changes. ([Theory: TF-IDF](../concepts.md#4-tf-idf))

---

## Before vs after (measured)

| `"is a kitten a good pet"` | Before (keyword) | After (TF-IDF) |
|---|---|---|
| 1st | `coffee.txt` 3 (tie, wins only alphabetically) | `felines.txt` 0.046 |
| 2nd | `felines.txt` 3 (tie) | `python_snake.txt` 0.021 |
| 3rd | `python_language.txt` 3 (tie) | `coffee.txt` 0.020 |

---

## 2. Code example

### 2.1 The formula as a loop
```python
total = 0.0                                   # start at 0 (a float, because tf × idf has decimals)
for word in query_words:
    total = total + tf(word, doc_words) * idf(word, all_doc_words)
```
That's the whole `score`. Each question word adds its weight.

### 2.2 Your keyword search vs. TF-IDF search
TF-IDF `search` is your keyword `search` from Step 4 with **one new line** and **one changed line**:

```python
# keyword search (Step 4)                    # TF-IDF search (this step)
results = []                                  results = []
for doc in docs:                              for doc in docs:
                                                  doc_words = tokenize(doc.text)              # ← NEW
    s = score(query=query, doc=doc)               s = score(query, doc_words, all_doc_words)   # ← CHANGED
    if s > 0:                                     if s > 0:
        results.append((s, doc))                      results.append((s, doc))
```

**Why is `all_doc_words` built above the loop, but `doc_words` inside?**

| Name | What it holds | Where |
|---|---|---|
| `all_doc_words` | the word lists of **all** documents, the same every time | **above** the loop, built once |
| `doc_words` | the words of **this one** document, different each round | **inside** the loop |

---

## 3. Goal

Add two functions to `tfidf.py`. Import `tokenize` from `ragi.retrieval.keyword` and `Document` from `ragi.loaders`.

**`score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float`**
1. Tokenize the query.
2. Add up `tf × idf` for each query word (the loop above).
3. Return the total.

**`search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]`**
1. **Before the loop**, build `all_doc_words = [tokenize(d.text) for d in docs]`.
2. `results = []`. For each doc: `doc_words = tokenize(doc.text)`, then `s = score(query, doc_words, all_doc_words)`.
3. If `s > 0`, append `(s, doc)`.
4. Sort and return the top k, exactly like your keyword `search`.

Then copy the `if __name__ == "__main__":` search box from `keyword.py` to the bottom of `tfidf.py`.

### ✅ Checkpoint
```bash
uv run python -m ragi.retrieval.tfidf
```
Type `is a kitten a good pet`:
```
0.046209812037329684 felines.txt
0.020548719460841487 python_snake.txt
0.019840142927709022 coffee.txt
```

Compare that with keyword search for the same question:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| **Keyword** | coffee.txt (3) ❌ | felines.txt (3) | python_language.txt (3) |
| **TF-IDF** | **felines.txt (0.046)** ✅ | python_snake.txt (0.021) | coffee.txt (0.020) |

Keyword search sees a **3-way tie**: all three files contain "is" and "a". `coffee.txt` wins only because it comes first alphabetically. TF-IDF knows "a" is worth 0 and "kitten" is rare (idf 1.386), so the cat document clearly wins.

Now try `how long can a python sleep`. TF-IDF ranks `python_language.txt` above `python_snake.txt`, just because it repeats "python" 4 times. Step 7 (BM25) fixes that.

---

## 4. Test

**This test checks:** a rare query word earns more than a common one. **How:** "a" is in all 3 documents (idf 0), "kitten" in only one, so the kitten doc must outscore the dog doc.
```python
def test_score_rewards_rare_words_over_common_ones():
    all_doc_words = [["a", "kitten"], ["a", "dog"], ["a", "cat"]]
    rare = score("a kitten", ["a", "kitten"], all_doc_words)
    common = score("a kitten", ["a", "dog"], all_doc_words)
    assert rare > common
```

**This test checks:** no shared words means a score of exactly 0. **How:** "python" appears in no document.
```python
def test_score_is_zero_without_shared_words():
    all_doc_words = [["coffee"], ["tea"]]
    assert score("python", ["coffee"], all_doc_words) == 0.0
```

**This test checks:** TF-IDF breaks the keyword tie on the real data. **How:** keyword search ties three files at 3; TF-IDF must put `felines.txt` first.
```python
def test_search_beats_keyword_tie_on_real_data():
    results = search("is a kitten a good pet", load_folder())
    assert results[0][1].source == "felines.txt"
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

Imports for these tests: `from ragi.loaders import Document, load_folder` and `from ragi.retrieval.tfidf import score, search`.

```bash
uv run pytest -v
```

---

## 5. Make it a module

Keyword `search` and TF-IDF `search` now take the **same inputs** `(query, docs, k)` and return the **same shape** (a list of `(score, Document)` pairs). That's on purpose: any later code can use either one, and swapping is a one-word change. BM25 and semantic search follow the same pattern.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/tfidf.py</code></strong></summary>

```python
import math
from ragi.loaders import Document
from ragi.retrieval.keyword import tokenize


def tf(word: str, docs_words: list[str]) -> float:
    if len(docs_words) == 0:
        return 0.0
    doc_words_len = len(docs_words)
    count = 0
    for doc_word in docs_words:
        if doc_word == word:
            count = count + 1
    return count / doc_words_len


def idf(word: str, all_doc_words: list[list[str]]) -> float:
    if len(all_doc_words) == 0:
        return 0.0

    n = len(all_doc_words)
    df = 0

    for single_doc in all_doc_words:
        if word in single_doc:
            df = df + 1

    if df == 0:
        return 0.0

    return math.log(n / df)


def score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float:
    tokenized_query = tokenize(query)
    total = 0.0
    for word in tokenized_query:
        total = total + tf(word, doc_words) * idf(word, all_doc_words)

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

</details>

---

**Previous:** [← Step 5: TF-IDF: the math](step-05-tfidf-the-math.md)

**Next:** [Step 7: BM25: the formula →](step-07-bm25-the-formula.md)
