---
title: "Step 7: BM25: the formula"
series: "Simple RAG"
step: 7
slug: "step-07-bm25-the-formula"
date: "2026-10-21"
tag: "RAG"
excerpt: "Upgrade TF-IDF to BM25, the ranking behind most real search engines: repeating a word stops paying off, and long documents get no free advantage."
minutes: 10
---

# Step 7: BM25: the formula

**You'll build:** `term_score()` and `score()` in `src/ragi/retrieval/bm25.py`.
**You'll learn:** saturation (`k1`), length fairness (`b`), `sum()`, and the difference between a total and an average.

---

## 1. Theory

**BM25** ("Best Matching 25") is the ranking behind Elasticsearch and most real search engines. It's TF-IDF with two fixes. **Saturation:** each repeat of a word counts for less than the one before, so spamming "python" 100 times can't win. **Length fairness:** a document longer than average gets a small penalty, and a shorter one a small boost. ([Theory: BM25](../concepts.md#5-bm25))

> **You don't need to memorize the formula.** Nobody writes BM25 from memory; they look it up. Remember the *idea* in one sentence, and keep the [cheat sheet](../concepts.md#cheat-sheet-you-dont-need-to-memorize-these) open.

### Fix 1: Saturation
How much a word counts when it appears `f` times:

| Times the word appears (f) | TF-style | BM25 |
|---|---|---|
| 1 | 1 | **1.00** |
| 2 | 2 | 1.43 |
| 4 | 4 | 1.82 |
| 10 | 10 | 2.17 |
| 100 | 100 | 2.46, and it can never pass 2.5 |

### Fix 2: Length fairness
One mention of a word, depending on document length (the average is 30 words):

| Document length | BM25 |
|---|---|
| 15 (short) | 1.29 ⬆ |
| 30 (average) | 1.00 |
| 60 (long) | 0.69 ⬇ |

---

## Before vs after (measured)

| `"python snake"` | Before (TF-IDF) | After (BM25) |
|---|---|---|
| `python_snake.txt` | 0.099 | 2.433 |
| `python_language.txt` | 0.087 (nearly tied, it repeats "python" 4×) | 1.241 |
| Gap between them | 1.1× | 2.0× |

---

## 2. Code example

The formula is the math-heavy part, so it's given. Put it in a new file, `src/ragi/retrieval/bm25.py`:

```python
def term_score(f: int, doc_len: int, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    length_ratio = doc_len / avg_len                                  # 1 = average length, 2 = twice as long
    return f * (k1 + 1) / (f + k1 * (1 - b + b * length_ratio))
```

| Name | Meaning |
|---|---|
| `f` | how many times the word appears in this document |
| `doc_len` / `avg_len` | this document's word count / the average word count of all documents |
| `k1 = 1.5` | the **saturation knob**: higher lets repetition count for more. Usually 1.2–2. (Not the same as `k` in `search`, which is the number of results.) |
| `b = 0.75` | the **length knob**: 0 ignores length, 1 applies the full penalty |
| `f * (k1 + 1) / (f + ...)` | `f` is on the top **and** the bottom, so the result flattens out as `f` grows. That's the saturation. |

### Two Python helpers you need
```python
sum([1, 2, 3])                     # 6 ← adds up a list of numbers
["a", "b", "a"].count("a")         # 2 ← how many times an item appears in a list
```

---

## 3. Goal

Add `score` to `bm25.py`, with the same signature as TF-IDF's:
```python
def score(query: str, doc_words: list[str], all_doc_words: list[list[str]]) -> float:
```
Imports: `tokenize` from `ragi.retrieval.keyword`, and **your own `idf` from `ragi.retrieval.tfidf`**. BM25 reuses it.

Steps:
1. If there are no documents (`len(all_doc_words) == 0`), return `0.0`.
2. Tokenize the query.
3. `avg_len` = **total** words of all documents **÷ number of documents**:
   `sum([len(w) for w in all_doc_words]) / len(all_doc_words)`
4. `total = 0.0`. For each query word:
   - `f = doc_words.count(word)`
   - add `idf(word, all_doc_words) * term_score(f, len(doc_words), avg_len)`
5. Return `total`.

It's your TF-IDF `score`, with `tf(...)` swapped for `term_score(...)`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.loaders import load_folder; from ragi.retrieval.keyword import tokenize; from ragi.retrieval.bm25 import score; docs = load_folder(); w = [tokenize(d.text) for d in docs]; [print(d.source, round(score('python snake', tokenize(d.text), w), 3)) for d in docs]"
```
Expected:
```
coffee.txt 0.0
felines.txt 0.0
python_language.txt 1.241
python_snake.txt 2.433
```
For "python snake", TF-IDF gave **0.099 vs. 0.087**, nearly a tie, because `python_language.txt` says "python" 4 times. BM25 gives **2.433 vs. 1.241**, so the snake document clearly wins.

---

## 4. Test

**This test checks:** repeating a word helps less and less (saturation). **How:** 10 mentions must score less than 10 × one mention, and never pass `k1 + 1`.
```python
def test_term_score_saturates():
    once = term_score(1, doc_len=30, avg_len=30)
    ten_times = term_score(10, doc_len=30, avg_len=30)
    assert ten_times < 10 * once
    assert ten_times < 1.5 + 1  # can never pass k1 + 1
```

**This test checks:** long documents get no free advantage. **How:** one mention in a 15-word doc must beat one mention in a 60-word doc.
```python
def test_term_score_penalizes_long_docs():
    short = term_score(1, doc_len=15, avg_len=30)
    long = term_score(1, doc_len=60, avg_len=30)
    assert short > long
```

**This test checks:** a missing word adds nothing. **How:** `f = 0` must give exactly 0.
```python
def test_term_score_is_zero_when_word_missing():
    assert term_score(0, doc_len=30, avg_len=30) == 0.0
```

**This test checks:** no shared words means 0, and no documents doesn't crash. **How:** the second call would divide by zero without the guard.
```python
def test_score_is_zero_without_shared_words():
    all_doc_words = [["coffee"], ["tea"]]
    assert score("python", ["coffee"], all_doc_words) == 0.0


def test_score_handles_no_documents():
    assert score("python", [], []) == 0.0
```

**This test checks:** the real numbers from the checkpoint. **How:** `approx(..., abs=0.001)` allows tiny rounding differences in decimal numbers.
```python
def test_score_matches_checkpoint_on_real_data():
    docs = load_folder()
    all_doc_words = [tokenize(d.text) for d in docs]
    scores = {d.source: score("python snake", tokenize(d.text), all_doc_words) for d in docs}
    assert scores["python_snake.txt"] == approx(2.433, abs=0.001)
    assert scores["python_language.txt"] == approx(1.241, abs=0.001)
```

Imports: `from pytest import approx`, `from ragi.loaders import load_folder`, `from ragi.retrieval.keyword import tokenize`, `from ragi.retrieval.bm25 import term_score, score`.

```bash
uv run pytest -v
```

---

## 5. Make it a module

`bm25.score` has the same inputs and output as `tfidf.score`. In Step 8 you'll give BM25 a `search` (copied from TF-IDF with **zero changes**) and compare all three methods side by side.

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/bm25.py</code></strong></summary>

```python
from ragi.retrieval.keyword import tokenize
from ragi.retrieval.tfidf import idf


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
```

</details>

---

**Previous:** [← Step 6: TF-IDF: search](step-06-tfidf-search.md)

**Next:** [Step 8: BM25: search and compare →](step-08-bm25-search-and-compare.md)
