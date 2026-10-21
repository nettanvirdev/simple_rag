---
title: "Step 3: Keyword search: tokenize and score"
series: "Simple RAG"
step: 3
slug: "step-03-keyword-tokenize-and-score"
date: "2026-10-21"
tag: "RAG"
excerpt: "Split text into words with a regular expression, then score a document by how many of the question's words it contains."
minutes: 15
---

# Step 3: Keyword search: tokenize and score

**You'll build:** `src/ragi/retrieval/keyword.py` with `tokenize()` and `score()`.
**You'll learn:** regular expressions, raw strings, and the difference between `in` on a string and `in` on a list.

---

## 1. Theory

Keyword search breaks the question and each document into words, then counts how many question words appear in each document. More matches means a higher score. It's fast and exact, but it can't understand meaning: searching "cat" won't find a document that only says "felines". More in [concepts: Keyword search](../concepts.md#3-keyword-search).

We'll build four search methods (keyword, TF-IDF, BM25, semantic), so they get their own folder:
```
src/ragi/retrieval/
├── __init__.py    ← empty file; makes this folder a package
└── keyword.py     ← this step
```

---

## Before vs after (measured)

| | Before (`word in doc.text`) | After (`word in tokenize(doc.text)`) |
|---|---|---|
| `score("cat", "we concatenate strings")` | 1 (wrong: "con**cat**enate") | 0 |
| `score("python", "Python is great")` | 0 (wrong: capital P) | 1 |
| Comparison | letters inside a string | whole lowercase words |

---

## 2. Code example

### 2.1 `tokenize`
**Tokenizing** means splitting text into words, called **tokens**.

```python
import re                                   # "regular expressions": a mini-language for finding patterns in text

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
```

| Piece | Meaning |
|---|---|
| `text.lower()` | `"Python"` → `"python"`, so uppercase and lowercase match |
| `r"\w+"` | the pattern: `\w` = a letter, digit or `_`; `+` = one or more in a row. Together: "a word" |
| `r"..."` | a *raw* string, which keeps the `\` as it is |
| `re.findall(...)` | finds every match and returns them as a list |

```python
tokenize("Hello, RAG world!")   # ['hello', 'rag', 'world']  ← commas and ! are gone
```

### 2.2 How to read a pattern
A pattern is built from small pieces:

| Piece | Means | Example match |
|---|---|---|
| `a` | the letter a, literally | `a` |
| `\w` | one **word character**: letter, digit or `_` | `t`, `7`, `_` |
| `\d` | one **digit** | `4` |
| `\s` | one **space** (or tab or newline) | ` ` |
| `\S` | one **non-space** character | `!`, `t` |
| `[abc]` | one of **these** characters | `a` or `b` or `c` |
| `[a-z]` | one character in this **range** | any lowercase letter |
| `+` | the previous piece, **1 or more** times | `\d+` → `1999` |
| `*` | the previous piece, **0 or more** times | `t\w*` → `t`, `tea` |
| `{4,}` | the previous piece, **4 or more** times | `\w{4,}` → `born` |

The same text through different patterns:
```python
text = "Hi! I'm Alex, born 1999. Email: alex@x.com #RAG_2026".lower()
```
| Pattern | Result | Use it when |
|---|---|---|
| `\w+` | `['hi', 'i', 'm', 'alex', 'born', '1999', 'email', 'alex', 'x', 'com', 'rag_2026']` | general words (our choice) |
| `[a-z]+` | `['hi', 'i', 'm', 'alex', 'born', 'email', 'alex', 'x', 'com', 'rag']` | letters only, numbers dropped |
| `\d+` | `['1999', '2026']` | extracting numbers |
| `[a-z']+` | `['hi', "i'm", 'alex', ...]` | keeping `i'm` as one word |
| `\S+` | `['hi!', "i'm", 'alex,', ..., 'alex@x.com', '#rag_2026']` | split on spaces only, punctuation kept |
| `\w{4,}` | `['alex', 'born', '1999', 'email', 'alex', 'rag_2026']` | dropping short words |
| `#\w+` | `['#rag_2026']` | finding hashtags |

No pattern is perfect: `\w+` breaks `i'm` into `i` and `m`. Choose based on your data. `\w+` also handles letters like `é` in `café`, while `[a-z]+` would cut it to `caf`.

### 2.3 Why `r"..."`
In a normal Python string, `\` starts a special character:
```python
len("\n")    # 1 → \n became ONE character: a newline
len(r"\n")   # 2 → raw string: a real \ and a real n
```
Regex needs the real `\`. Without `r`, things break silently:
```python
re.findall("\bcat\b",  "a cat sat")   # []       ❌ Python turned \b into a "backspace" character
re.findall(r"\bcat\b", "a cat sat")   # ['cat']  ✅ regex receives \b = "edge of a word"
```
**Rule:** always write regex patterns as `r"..."`.

### 2.4 `in` on a string vs. a list
```python
"cat" in "we concatenate strings"              # True  ← looks inside the letters
"cat" in ["we", "concatenate", "strings"]      # False ← compares whole items
```
Keyword search must compare **whole words**, so both sides need to be tokenized lists.

---

## 3. Goal

Create `src/ragi/retrieval/__init__.py` (empty) and `src/ragi/retrieval/keyword.py` with `tokenize` from above, then add:

```python
def score(query: str, doc: Document) -> int:
```
You need `from ragi.loaders import Document` at the top. Steps:
1. `query_words` = tokenize the query.
2. `doc_words` = tokenize `doc.text`. **Don't skip this step.**
3. `count = 0`
4. Loop over `query_words`: if the word is `in doc_words`, add 1 to `count`.
5. Return `count`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.loaders import load_folder; from ragi.retrieval.keyword import score; docs = load_folder(); [print(d.source, score('python snake', d)) for d in docs]"
```
Expected:
```
coffee.txt 0
felines.txt 0
python_language.txt 1
python_snake.txt 2
```
Keyword search can't tell the two pythons apart, but counting helps.

---

## 4. Test

You don't need `tmp_path` here, because a test can build a `Document` directly.

**This test checks:** messy text becomes clean lowercase words. **How:** if `.lower()` were missing, `"RAG"` would stay uppercase.

```python
from ragi.loaders import Document
from ragi.retrieval.keyword import tokenize, score


def test_tokenize_lowercases_and_drops_punctuation():
    data = tokenize("Hello, RAG!")
    assert data[0] == "hello"
    assert data[1] == "rag"
```

**This test checks:** each question word found in the document adds 1. **How:** both "python" and "snake" appear, so the score is 2.

```python
def test_score_counts_matching_words():
    doc = Document(text="the python is a snake", source="x.txt")
    scores = score("python snake", doc)
    assert scores == 2
```

**This test checks:** keyword search's weakness, that it can't match meaning. **How:** "cat" isn't in "felines purr", so the score is 0. Semantic search fixes this in Step 9.

```python
def test_score_is_zero_for_synonyms():
    doc = Document(text="felines purr", source="x.txt")
    scores = score("cat", doc)
    assert scores == 0
```

**This test checks:** "python" matches "Python". **How:** the document is capitalized and the query isn't.

```python
def test_score_ignores_case():
    doc = Document(text="Python is great", source="x.txt")
    scores = score("python", doc)
    assert scores == 1
```

**This test checks:** a word hidden inside a longer word doesn't count. **How:** "cat" is inside "con**cat**enate", but it isn't a word there.

```python
def test_score_matches_whole_words_only():
    doc = Document(text="we concatenate strings", source="x.txt")
    scores = score("cat", doc)
    assert scores == 0
```

```bash
uv run pytest -v
```

**Try TDD (test-driven development):** write the last two tests *before* fixing a buggy `score`, watch them go red, then fix the code and watch them turn green.

---

## 5. Make it a module

Step 4 adds `search()` to this same file, calling `score()` on every document. Later, TF-IDF and BM25 reuse your tokenizer:
```python
from ragi.retrieval.keyword import tokenize
```
Write it once, use it everywhere.

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
```

</details>

---

**Previous:** [← Step 2: Loading text files](step-02-loading-text-files.md)

**Next:** [Step 4: Keyword search: ranking →](step-04-keyword-search.md)
