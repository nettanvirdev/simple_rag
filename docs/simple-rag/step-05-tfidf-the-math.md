---
title: "Step 5: TF-IDF: the math"
series: "Simple RAG"
step: 5
slug: "step-05-tfidf-the-math"
date: "2026-10-21"
tag: "RAG"
excerpt: "Give every word a weight: term frequency says how much of a document a word is, inverse document frequency says how rare it is everywhere."
minutes: 15
---

# Step 5: TF-IDF: the math

**You'll build:** `src/ragi/retrieval/tfidf.py` with `tf()` and `idf()`.
**You'll learn:** term frequency, inverse document frequency, why `log`, `list.count()`, `math.log`, and comparing decimals in tests with `approx`.

---

## 1. Theory

Keyword search counts every word equally, so "a" is worth as much as "espresso". **TF-IDF** gives each word a **weight**. **TF** (term frequency) asks how often the word appears in *this* document. **IDF** (inverse document frequency) asks how *rare* it is across *all* documents. Multiply them, and a word scores high when it's frequent here but rare elsewhere. More in [concepts: TF-IDF](../concepts.md#4-tf-idf).

---

## Before vs after (measured)

| Word | Before (keyword search) | After (IDF weight) |
|---|---|---|
| `a` (in 4 of 4 docs) | 1 | 0.000 |
| `is` (in 3 of 4) | 1 | 0.288 |
| `python` (in 2 of 4) | 1 | 0.693 |
| `coffee` (in 1 of 4) | 1 | 1.386 |

---

## 2. Code example

### 2.1 TF: how much of this document is this word?
```
tf = (times the word appears in the doc) / (total words in the doc)
```
In `coffee.txt`, "coffee" appears 3 times out of 29 words, so **tf = 3/29 = 0.103**. Dividing by the length keeps it fair: a long document doesn't win just by having more words.

### 2.2 IDF: how rare is this word?
```
idf = log(N / df)        N  = total number of documents (4)
                         df = how many documents contain the word
```
Real numbers from the four practice files:

| Word | Found in (df) | N/df | idf | Meaning |
|---|---|---|---|---|
| `a` | 4 of 4 | 1 | **0.000** | in every doc, so it tells you nothing |
| `is` | 3 of 4 | 1.33 | 0.288 | almost useless |
| `python` | 2 of 4 | 2 | 0.693 | somewhat useful |
| `coffee` | 1 of 4 | 4 | **1.386** | rare, so it's a strong signal |

"a" now counts as **zero**. Nobody had to write a list of boring words; the math found them.

### 2.3 Why `log`?
`log` shrinks big numbers so rare words don't explode:

| N/df | log |
|---|---|
| 1 | 0 |
| 2 | 0.69 |
| 10 | 2.3 |
| 1,000,000 | 13.8 |

Without the log, a word found in 1 of a million documents would be worth a million times more than anything else. With it, it's worth 13.8: still high, but under control. Python does the math for you.

### 2.4 The Python pieces
```python
import math                                  # Python's built-in math tools

math.log(4)                                  # 1.386... (the "natural log")
["a", "b", "a"].count("a")                   # 2 ← how many times an item appears in a list
len(["a", "b", "a"])                         # 3 ← how many items in total
```

---

## 3. Goal

Create `src/ragi/retrieval/tfidf.py` with two functions. Both work on **lists of words** (already tokenized), not on Documents.

**`tf(word: str, doc_words: list[str]) -> float`** (`float` = a number with decimals, like `0.103`)
1. If `doc_words` is empty, return `0.0` (dividing by 0 would crash).
2. Count how many items equal `word`: either a loop **with an `if`**, or simply `doc_words.count(word)`.
3. Return the count divided by `len(doc_words)`.

**`idf(word: str, all_doc_words: list[list[str]]) -> float`**
`list[list[str]]` is a list of word lists, one per document:
```python
[["coffee", "is", "brewed", ...], ["felines", "are", ...], ...]
```
1. `n` = number of documents (`len`).
2. `df = 0`. Loop over the documents: **if `word` is `in` that document's list, add 1**. No inner loop needed; `in` already checks the whole list. df counts **documents**, not words.
3. If `df == 0`, return `0.0` (the word appears nowhere).
4. Otherwise return `math.log(n / df)`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.loaders import load_folder; from ragi.retrieval.keyword import tokenize; from ragi.retrieval.tfidf import tf, idf; w = [tokenize(d.text) for d in load_folder()]; print(tf('a', ['a','b','a','c']), tf('zebra', ['a','b','a','c'])); print(idf('a', w), idf('python', w), idf('coffee', w), idf('zebra', w))"
```
Expected:
```
0.5 0.0
0.0 0.6931471805599453 1.3862943611198906 0.0
```
An idf can never be negative. If you see the same negative number for every word, your df is counting words instead of documents.

---

## 4. Test

One new idea: decimals are slightly inexact in computers (`0.1 + 0.2` gives `0.30000000000000004`), so for them we use pytest's **`approx(...)`**: "equal, give or take a tiny rounding error".

**This test checks:** tf is count divided by length. **How:** "a" is 2 of 4 words, so 0.5.

```python
import math

from pytest import approx

from ragi.retrieval.tfidf import tf, idf


def test_tf_is_count_divided_by_length():
    assert tf("a", ["a", "b", "a", "c"]) == 0.5
```

**This test checks:** a missing word, and an empty document, give 0 instead of a wrong number or a crash. **How:** "zebra" isn't there, and `[]` has no words.

```python
def test_tf_is_zero_for_missing_word():
    assert tf("zebra", ["a", "b"]) == 0.0


def test_tf_handles_empty_doc():
    assert tf("a", []) == 0.0
```

**This test checks:** a word in every document is worth nothing. **How:** "a" is in both, so log(2/2) = 0.

```python
def test_idf_is_zero_for_word_in_every_doc():
    all_doc_words = [["a", "cat"], ["a", "dog"]]
    assert idf("a", all_doc_words) == 0.0
```

**This test checks:** the result matches the `log(N/df)` formula. **How:** "python" is in 2 of 4 documents.

```python
def test_idf_matches_formula():
    all_doc_words = [["python"], ["python"], ["coffee"], ["tea"]]
    assert idf("python", all_doc_words) == approx(math.log(4 / 2))
```

**This test checks:** rare words weigh more. **How:** "coffee" is in 1 document and "python" in 3.

```python
def test_idf_is_higher_for_rare_words():
    all_doc_words = [["python", "coffee"], ["python"], ["python"], ["tea"]]
    assert idf("coffee", all_doc_words) > idf("python", all_doc_words)
```

**This test checks:** df counts documents, not occurrences. **How:** "python" appears 3 times but in only 1 of 2 documents, so idf = log(2/1).

```python
def test_idf_counts_documents_not_occurrences():
    all_doc_words = [["python", "python", "python"], ["tea"]]
    assert idf("python", all_doc_words) == approx(math.log(2 / 1))
```

**This test checks:** an unknown word gives 0 instead of a division-by-zero crash.

```python
def test_idf_is_zero_for_unknown_word():
    assert idf("zebra", [["a"], ["b"]]) == 0.0
```

```bash
uv run pytest -v
```

---

## 5. Make it a module

Step 6 adds `score()` and `search()` to this same file. `score` adds up `tf × idf` for each word in the question, and `search` ranks documents with it, the same way keyword `search` does.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/tfidf.py</code></strong></summary>

```python
import math


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
```

</details>

---

**Previous:** [← Step 4: Keyword search: ranking](step-04-keyword-search.md)

**Next:** [Step 6: TF-IDF: search →](step-06-tfidf-search.md)
