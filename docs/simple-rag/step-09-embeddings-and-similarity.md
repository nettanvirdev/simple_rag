---
title: "Step 9: Embeddings and similarity"
series: "Simple RAG"
step: 9
slug: "step-09-embeddings-and-similarity"
date: "2026-10-21"
tag: "RAG"
excerpt: "Turn text into 384 numbers that capture meaning, and measure how alike two texts are with cosine similarity, even when they share no words."
minutes: 12
---

# Step 9: Embeddings and similarity

**You'll build:** `embed()` and `similarity()` in `src/ragi/retrieval/semantic.py`.
**You'll learn:** what an embedding is, cosine similarity, `shape`, tensors, and `.item()`.

---

## 1. Theory

An **embedding model** turns text into a list of numbers (a **vector**) that represents its **meaning**. Texts with similar meaning get similar numbers, even when they share no words. **Cosine similarity** measures how alike two vectors are: 1 means the same meaning, 0 means unrelated. ([Theory: semantic search](../concepts.md#6-semantic-search))

The model used here, `all-MiniLM-L6-v2`, turns any text into **384 numbers**:
```
"cat" → [0.037, 0.051, -0.0, 0.06, -0.117, ... 384 numbers in total]
```
No single number means "furry" or "animal". The meaning is spread across all 384.

Similarities measured with this model:

| Text A | Text B | Similarity |
|---|---|---|
| cat | kitten | **0.788** |
| cat | feline | **0.698** |
| cat | coffee | 0.429 |
| python snake | a large reptile | **0.439** |
| python snake | programming language | 0.261 |

"cat" and "kitten" share no words, yet they're the closest pair. Keyword search, TF-IDF and BM25 could never do that.

### The idea with 2 numbers instead of 384
```
cat    = [0.9, 0.1]   ↗ points almost the same way
kitten = [0.8, 0.2]   ↗
coffee = [0.1, 0.9]   ↑ points a different way
```
Cosine similarity checks **how much two arrows point in the same direction**. Worked out by hand, cat vs. kitten is 0.99 and cat vs. coffee is 0.22. The library does this math for you.

---

## Before vs after (measured)

| | Before (word matching) | After (cosine similarity of embeddings) |
|---|---|---|
| `cat` vs `kitten` | 0 shared words | 0.788 |
| `cat` vs `coffee` | 0 shared words | 0.429 |
| `python snake` vs `a large reptile` | 0 shared words | 0.439 |

---

## 2. Code example

### 2.1 Install
```bash
uv add sentence-transformers
```
The first time you load the model, it downloads about 90 MB.

### 2.2 Explore in the terminal first
```bash
uv run python
```
```python
from sentence_transformers import SentenceTransformer, util   # the library we just installed
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")   # loads the model (a few seconds)

v = model.encode("cat")          # text → vector
v.shape                           # (384,) ← 384 numbers
v[:5]                             # the first 5

util.cos_sim(model.encode("cat"), model.encode("kitten"))          # tensor([[0.7882]])
util.cos_sim(model.encode("cat"), model.encode("kitten")).item()   # 0.7882 ← .item() pulls out the plain number
```
You can ignore the warnings about **symlinks on Windows** and **HF_TOKEN**. They're harmless.

### 2.3 Common questions
1. **Why `(384,)` with nothing after the comma?** `shape` is a tuple, and a one-item tuple needs a comma: `(384)` would just be the number 384. One number in the shape means one direction: a single row of 384 numbers. A table of 4 documents would be `(4, 384)`.
2. **Is `shape` "how many numbers are inside"?** Yes: the size in each direction. `(384,)` is one vector of 384 numbers.
3. **Does `cos_sim` calculate distance?** It calculates **similarity**: higher = more alike (1 = same meaning, 0 = unrelated). Distance is the opposite, `1 - similarity`.
4. **What is `tensor([[0.7882]])`, and why so many brackets?** `tensor` is PyTorch's name for an array of numbers (the library runs on PyTorch). `[[ ]]` is a **table**: the outer brackets hold the rows, the inner brackets are one row. `cos_sim` always returns a table with one row per text in A and one column per text in B. One text vs. one text gives a 1×1 table.
5. **How does `.item()` get the plain number?** It's a tensor function meaning "I hold exactly one number; here it is as a normal Python number". It unwraps `tensor([[0.7882]])` into `0.7882`. With more than one number inside, it raises an error.

---

## 3. Goal

Create `src/ragi/retrieval/semantic.py` starting with:
```python
from sentence_transformers import SentenceTransformer, util
from ragi import config

model = SentenceTransformer(config.EMBEDDING_MODEL)   # load once, when the file is imported
```
Then write two small functions:
- **`embed(text: str)`**: returns `model.encode(text)`. One line, and remember the `return`.
- **`similarity(a: str, b: str) -> float`**: embeds both texts and returns `util.cos_sim(...)` with `.item()`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.retrieval.semantic import similarity; print(round(similarity('cat', 'kitten'), 3), round(similarity('cat', 'coffee'), 3))"
```
Expected: `0.788 0.429`

---

## 4. Test

**This test checks:** the model produces 384 numbers per text. **How:** the vector's `shape` must be `(384,)`.
```python
def test_embed_returns_384_numbers():
    assert embed("cat").shape == (384,)
```

**This test checks:** a text is identical to itself. **How:** cosine similarity of "cat" with "cat" must be 1.0, give or take rounding.
```python
def test_text_is_identical_to_itself():
    assert similarity("cat", "cat") == approx(1.0, abs=0.001)
```

**This test checks:** similar meanings score higher. **How:** "kitten" shares no words with "cat" but must still beat "coffee".
```python
def test_similar_meaning_scores_higher():
    assert similarity("cat", "kitten") > similarity("cat", "coffee")
```

Imports: `from pytest import approx` and `from ragi.retrieval.semantic import embed, similarity`.

```bash
uv run pytest -v
```
These tests load the real model, so they take a few seconds.

---

## 5. Make it a module

In Step 10, `search(query, docs, k)` embeds every document, embeds the question, and ranks by similarity, with the same shape as your other three searches. Then it joins `compare.py`. A preview:

| Question | Keyword / TF-IDF / BM25 | Semantic |
|---|---|---|
| `cat` | nothing found | **felines.txt** (0.459) |
| `what helps me stay awake` | nothing useful | **coffee.txt** (0.356) |
| `a reptile that squeezes its prey` | weak | **python_snake.txt** (0.398) |

---

<details>
<summary><strong>Show solution: <code>src/ragi/retrieval/semantic.py</code></strong></summary>

```python
from sentence_transformers import SentenceTransformer, util
from ragi import config

model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(text: str):
    return model.encode(text)


def similarity(a: str, b: str) -> float:
    similarity = util.cos_sim(embed(a), embed(b)).item()
    return similarity
```

</details>

---

**Previous:** [← Step 8: BM25: search and compare](step-08-bm25-search-and-compare.md)

**Next:** [Step 10: Semantic search →](step-10-semantic-search.md)
