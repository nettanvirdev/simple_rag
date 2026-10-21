---
title: "Step 4: Keyword search: ranking"
series: "Simple RAG"
step: 4
slug: "step-04-keyword-search"
date: "2026-10-21"
tag: "RAG"
excerpt: "Turn a score into search results: score every document, drop the zeros, sort best first, and keep the top k, then search from your terminal."
minutes: 15
---

# Step 4: Keyword search: ranking

**You'll build:** `search()` in `src/ragi/retrieval/keyword.py`, plus an interactive search box.
**You'll learn:** tuples, sorting with `key=lambda`, slicing, unpacking, and `if __name__ == "__main__"`.

---

## 1. Theory

A score alone isn't search. Search means scoring **every** document, keeping the ones that match, sorting them best first, and returning only the top **k**. In RAG, the top k is what gets handed to the LLM. More in [concepts: From scores to search results](../concepts.md#3-keyword-search).

---

## Before vs after (measured)

| | Before (score only) | After (`search`) |
|---|---|---|
| Output | one number for one document | ranked top-k list over all documents |
| `"python snake"` | — | `python_snake.txt` (2), `python_language.txt` (1) |
| `"cat"` | — | no results (no document contains the word) |

---

## 2. Code example

Try each idea in the playground first (`uv run python`).

### 2.1 Tuples
A **tuple** is a few values packed together in round brackets. Here, a (score, name) pair:
```python
pair = (3, "snake")
pair[0]    # 3        ← first item
pair[1]    # 'snake'  ← second item
```

### 2.2 Sorting by a chosen part: `key=lambda`
```python
pairs = [(1, "coffee"), (3, "snake"), (2, "code")]
sorted(pairs, key=lambda p: p[0], reverse=True)
# [(3, 'snake'), (2, 'code'), (1, 'coffee')]
```

**A lambda is just a function without a name.** These two do exactly the same thing:
```python
def get_score(pair):        # normal function, with a name
    return pair[0]

lambda pair: pair[0]        # the same function, in one line, no name
```
Read `lambda pair: pair[0]` as: `pair` is the input (before `:`), and `pair[0]` is what it returns (after `:`).

**What `key=` does:** your list holds pairs like `(2, Document(...))`, and Python can't decide on its own which pair is "bigger". `key=` says: *"call this function on each item, and sort by what it returns."*

| Item | key function returns | |
|---|---|---|
| `(1, doc_code)` | `1` | |
| `(2, doc_snake)` | `2` | ← biggest, so it comes first (`reverse=True`) |

`reverse=True` means highest first. By default, sorting goes lowest first.

You could skip the lambda entirely; this works the same:
```python
sorted(results, key=get_score, reverse=True)   # pass the function itself, without ()
```
A lambda just saves writing a separate `def` for a one-line job. More practice:
```python
words = ["banana", "kiwi", "apple"]
sorted(words)                                  # alphabetical: apple, banana, kiwi
sorted(words, key=lambda w: len(w))            # by length: kiwi, apple, banana
sorted(words, key=lambda w: w[-1])             # by LAST letter: banana, apple, kiwi
```
Each lambda answers one question: "what should I compare for each item?"

### 2.3 Slicing: take the first few items
```python
top = [(3, 'snake'), (2, 'code'), (1, 'coffee')]
top[:2]            # [(3, 'snake'), (2, 'code')]  ← from the start, up to (not including) index 2
[10, 20, 30][:5]   # [10, 20, 30]  ← asking for more than exists is fine
```

### 2.4 Unpacking: split a tuple into two variables
```python
for s, name in top:
    print(s, name)     # 3 snake / 2 code / 1 coffee
```

---

## 3. Goal

Add this to `keyword.py`:
```python
def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[int, Document]]:
```
`list[tuple[int, Document]]` means "a list of (score, Document) pairs".

Steps:
1. `results = []`
2. Loop over `docs`; for each, `s = score(query, doc)`.
3. If `s > 0`, append the pair `(s, doc)` to `results`. Documents with no matching words are dropped.
4. Sort `results` by score, highest first (`key=lambda` and `reverse=True`).
5. Return the first `k` with `[:k]`.

Then add an interactive search box at the **bottom** of `keyword.py`:
```python
if __name__ == "__main__":                # only runs when you start this file directly, not when another file imports it
    from ragi.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
```

### ✅ Checkpoint
`-m` runs a module by its import name. Press `Ctrl+C` to quit.
```bash
uv run python -m ragi.retrieval.keyword
```
```
search> python snake
2 python_snake.txt
1 python_language.txt
search> cat
search>
```
`cat` returns nothing, even though `felines.txt` is about cats. That's the problem later steps fix.

---

## 4. Test

**This test checks:** the document with the most matching words comes first. **How:** the weaker document (1 match) is first in the list, so only real sorting can put the snake document (2 matches) on top.

```python
from ragi.loaders import Document
from ragi.retrieval.keyword import search


def test_search_ranks_best_match_first():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="python snake", source="snake.txt"),
    ]
    results = search("python snake", docs)
    assert results[0][1].source == "snake.txt"     # results[0] = best pair, [1] = its Document
```

**This test checks:** documents with no matching words aren't returned. **How:** "coffee beans" shares nothing with "python".

```python
def test_search_drops_zero_scores():
    docs = [
        Document(text="python code", source="code.txt"),
        Document(text="coffee beans", source="coffee.txt"),
    ]
    results = search("python", docs)
    assert len(results) == 1
```

**This test checks:** you never get more than `k` results. **How:** 3 documents match, but `k=2`.

```python
def test_search_returns_at_most_k():
    docs = [Document(text="python", source=f"{i}.txt") for i in range(3)]
    results = search("python", docs, k=2)
    assert len(results) == 2
```

```bash
uv run pytest -v
```

---

## 5. Make it a module

In Step 12, the RAG pipeline calls a `search(question, docs)` and sends the top documents to the LLM. Every search method in this series (TF-IDF, BM25, semantic) keeps this exact shape, `search(query, docs, k)` returning `(score, Document)` pairs, so they're interchangeable.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/keyword.py</code></strong></summary>

```python
import re
from ragi.loaders import Document


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def score(query: str, doc: Document) -> int:
    query_words = tokenize(query)
    count = 0
    tokenized_doc_text = tokenize(doc.text)
    for word in query_words:
        if word in tokenized_doc_text:
            count = count + 1

    return count


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[int, Document]]:
    results = []

    for doc in docs:
        s = score(query=query, doc=doc)
        if s > 0:
            results.append((s, doc))

    sorted_results = sorted(results, key=lambda pair: pair[0], reverse=True)
    return sorted_results[:k]


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

**Previous:** [← Step 3: Keyword search: tokenize and score](step-03-keyword-tokenize-and-score.md)

**Next:** [Step 5: TF-IDF: the math →](step-05-tfidf-the-math.md)
