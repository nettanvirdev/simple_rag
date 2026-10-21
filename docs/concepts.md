---
title: "RAG Concepts"
series: "Simple RAG"
step: 0
slug: "concepts"
date: "2026-10-21"
tag: "RAG"
excerpt: "The theory and math behind keyword search, TF-IDF, BM25, embeddings, semantic search and RAG, with real numbers and a one-page cheat sheet."
minutes: 20
---

# RAG Concepts

The ideas behind Retrieval-Augmented Generation, explained from zero, with the math and real numbers.
This is the theory reference for the [step-by-step guide](simple-rag/README.md). The code for every idea lives in `src/ragi/`.

## Contents
- [What is RAG?](#what-is-rag)
- [1. Documents and loading](#1-documents-and-loading)
- [2. Tokenizing](#2-tokenizing)
- [3. Keyword search](#3-keyword-search)
- [4. TF-IDF](#4-tf-idf)
- [5. BM25](#5-bm25)
- [6. Semantic search](#6-semantic-search)
- [7. Simple RAG](#7-simple-rag)
- [Glossary](#glossary)

---

## Cheat sheet (you don't need to memorize these)
Remember the **idea**. Look up the formula here when you need it. Nobody in industry writes BM25 from memory.

| Concept | The idea in one sentence | Formula |
|---|---|---|
| **Document** | every file becomes "text + where it came from" | `Document(text, source)` |
| **Tokenize** | split text into lowercase words | `re.findall(r"\w+", text.lower())` |
| **Keyword search** | count shared words | score = number of query words in the doc |
| **TF** | how much of *this* doc is this word | count ÷ doc length |
| **IDF** | how *rare* the word is across all docs | log(N ÷ df) |
| **TF-IDF** | frequent here + rare elsewhere = important | Σ tf × idf |
| **BM25** | TF-IDF where repeating stops paying off and long docs get no free advantage | Σ idf × f(k1+1) / (f + k1(1 − b + b·len/avg)) |
| **Embedding** | text → a list of numbers that captures meaning | `model.encode(text)` → 384 numbers |
| **Cosine similarity** | do two vectors point the same way? 1 = same meaning, 0 = unrelated | A·B ÷ (‖A‖·‖B‖) |
| **Semantic search** | rank documents by cosine similarity to the question | always returns k results |
| **Search** | score all → drop 0 → sort high→low → keep top k | same for every method |
| **RAG** | retrieve top docs → paste into prompt → LLM answers only from them | `ask(context + question, system=RULES)` |

**Memory trick:** every method answers the same question, "how well does this document match?". Each one fixes the previous one's weakness:
```
keyword  ──"a" counts as much as "kitten"──►  TF-IDF  ──repeating a word keeps winning──►  BM25  ──can't understand meaning──►  semantic search
```

---

## What is RAG?

A Large Language Model (LLM) only knows what it saw during training. It doesn't know your CV, your company's documents, or anything newer than its training data. Ask it about them and it will guess, and sometimes make things up (this is called **hallucination**).

**RAG** fixes this in two steps:

1. **Retrieval**: find the pieces of *your* documents that are relevant to the question.
2. **Augmented Generation**: paste those pieces into the prompt and ask the LLM to answer **using only them**.

```
question ──► [ search your documents ] ──► top matching pieces
                                                │
                                                ▼
            answer ◄── [ LLM ] ◄── prompt = question + matching pieces
```

The LLM is good at *writing* the answer. Retrieval decides *what it knows* while writing it. So **most of RAG quality comes from retrieval**, and that's why the first sections of this course are all about search.

---

## 1. Documents and loading

Before we can search anything, every file (txt, pdf, html, ...) is turned into one common shape:

```python
@dataclass
class Document:
    text: str     # the content
    source: str   # where it came from, e.g. "coffee.txt"
```

**Why one shape?** Everything after loading (search, chunking, the LLM prompt) only ever sees `Document`s. Adding PDF support later means writing one new loader; nothing downstream changes.

**Why keep `source`?** So the final answer can cite where it came from ("according to coffee.txt..."). Citations let a user check the answer instead of trusting it blindly.

Code: `src/ragi/loaders.py`

---

## 2. Tokenizing

Computers can't compare "meaning" directly. The simplest trick is to compare **words**. **Tokenizing** splits text into a list of words, called **tokens**:

```python
tokenize("Hello, RAG world!")   # ['hello', 'rag', 'world']
```

We use a **regular expression** (regex), a small pattern language for finding text:

```python
re.findall(r"\w+", text.lower())
```

- `.lower()` makes "Python" and "python" the same token.
- `\w+` means "one or more letters/digits/underscores in a row", i.e. a word. Punctuation is dropped.

**Every tokenizer is a choice.** `\w+` turns `"I'm"` into `i`, `m` and `"alex@x.com"` into `alex`, `x`, `com`. Other patterns make other trade-offs. There's no perfect one, only one that fits your data.

> In Phase 2 we'll meet a different kind of tokenizer: LLMs and embedding models split text into *sub-word* pieces (`"unbelievable"` → `un`, `believ`, `able`). Same idea, finer cuts.

Code: `tokenize()` in `src/ragi/retrieval/keyword.py`

---

## 3. Keyword search

### The idea
Score each document by **how many of the question's words it contains**. More shared words = more relevant.

```
question: "python snake"
"the python is a snake"   → python ✓  snake ✓  → score 2
"python code"             → python ✓  snake ✗  → score 1
"coffee beans"            →                    → score 0
```

### From scores to search results
1. **Score** every document.
2. **Drop** documents scoring 0 (no shared words at all).
3. **Rank**: sort highest score first.
4. **Top-k**: keep only the best `k` (e.g. 3).

The **top-k** is what RAG sends to the LLM. The LLM can only read so much, so retrieval has to choose.

### Words, not letters
Matching must compare **whole tokens**, not raw text. Checking `"cat" in "we concatenate strings"` is `True` (the letters c-a-t are inside), but `"cat"` is not one of the *words*. Tokenize both sides, then compare.

### Limits (what the next methods fix)
| Problem | Example | Fixed by |
|---|---|---|
| Every word counts the same | `"a"` and `"is"` count as much as `"espresso"` | TF-IDF (§4 below) |
| Long documents win by size | A huge document contains more words, so it matches more question words by chance | TF (length normalization), then BM25 |
| No understanding of meaning | `"cat"` can't find a document about `"felines"` | Semantic search |
| Word forms differ | `"pythons"` ≠ `"python"` | Stemming / semantic search |

Code: `score()` and `search()` in `src/ragi/retrieval/keyword.py`

---

## 4. TF-IDF

**TF-IDF = Term Frequency × Inverse Document Frequency.** It gives each word a *weight* instead of counting every word as 1.

The intuition: a word is a strong signal for a document when it is **frequent in that document** but **rare in the collection**.

All numbers below come from the 4 practice files in `data/`.

### TF: Term Frequency
*"How much of this document is this word?"*

$$
\text{tf}(w, d) = \frac{\text{times } w \text{ appears in } d}{\text{total words in } d}
$$

`coffee.txt` has 29 words and says "coffee" 3 times → tf = 3/29 = **0.103**.

Dividing by the length is what keeps it fair: without it, a 10,000-word document would beat a focused 30-word one just by being long.

### IDF: Inverse Document Frequency
*"How rare is this word across all documents?"*

$$
\text{idf}(w) = \log\frac{N}{\text{df}(w)}
$$

- **N** = number of documents (here 4)
- **df** (document frequency) = how many documents contain the word at least once

| Word | df | N/df | idf | Reading |
|---|---|---|---|---|
| `a` | 4 | 1 | **0.000** | in every document, so it can't tell them apart |
| `is` | 3 | 1.33 | 0.288 | nearly useless |
| `python` | 2 | 2 | 0.693 | useful |
| `snake` | 1 | 4 | **1.386** | rare, so a strong signal |

A word found everywhere gets idf = log(1) = **0**: it is automatically ignored. Nobody had to write a list of "boring words". The math discovered them.

### Why the log?
`log` squashes big ratios so rare words don't take over:

| N/df | without log | with log |
|---|---|---|
| 1 | 1 | 0 |
| 2 | 2 | 0.69 |
| 10 | 10 | 2.3 |
| 1,000,000 | 1,000,000 | 13.8 |

Without it, a word in 1 of a million documents would be a million times more important than a common one. With it, it's important, but not absurdly so.

### TF-IDF score of a document
For each word in the question, multiply its tf in the document by its idf, then add them up:

$$
\text{score}(q, d) = \sum_{w \in q} \text{tf}(w, d) \times \text{idf}(w)
$$

### Worked example: `"is a python snake"`

| Document | Keyword score | TF-IDF score | What happened |
|---|---|---|---|
| python_snake.txt | 4 | **0.120** | `python` and `snake` both carry weight |
| python_language.txt | 3 | 0.105 | says "python" 4× in 32 words, a high tf |
| coffee.txt | 2 | 0.020 | only matched `is`, which is nearly worthless |
| felines.txt | 1 | **0.000** | only matched `a`, which is worth exactly 0 |

Keyword search gave coffee.txt a 2 and felines.txt a 1 just for containing "is" and "a". TF-IDF pushes them to almost nothing. That's the improvement.

### A tie broken: `"is a kitten a good pet"`

| Rank | Keyword search | TF-IDF |
|---|---|---|
| 1 | coffee.txt **3** (tie) | **felines.txt 0.046** |
| 2 | felines.txt **3** (tie) | python_snake.txt 0.021 |
| 3 | python_language.txt **3** (tie) | coffee.txt 0.020 |

All three tie for keyword search because they contain "is" and "a". The tie is then broken alphabetically, which is meaningless. TF-IDF sees that "kitten" is rare (idf 1.386) while "a" is worth 0, so the cat document clearly wins.

Notice the weakness too: python_language.txt nearly catches up just by **repeating "python" four times**. TF keeps growing with every repeat. BM25 (next) fixes that.

### Limits
- **Repetition keeps paying off.** Saying a word 10× counts 10× as much. → BM25 adds *saturation*.
- **Still no meaning.** `"cat"` still finds nothing about `"felines"`. → Semantic search.
- **IDF depends on the collection.** Add documents and every idf changes. That's fine, but scores aren't comparable across different collections.

Code: `tf()`, `idf()`, `score()` and `search()` in `src/ragi/retrieval/tfidf.py`

---

## 5. BM25

**BM25 ("Best Matching", attempt #25)** is TF-IDF with two fixes. It's the default ranking in Elasticsearch, OpenSearch and Lucene, and it's still a strong baseline that modern RAG systems combine with semantic search ("hybrid search", Phase 2).

### Fix 1: Saturation (knob `k1`)
In TF-IDF, saying a word 10× counts 10× as much. In BM25, each repeat adds **less** than the one before, and the total can never pass `k1 + 1`.

| Times the word appears (f) | TF-style | BM25 (k1 = 1.5) |
|---|---|---|
| 1 | 1 | 1.00 |
| 2 | 2 | 1.43 |
| 4 | 4 | 1.82 |
| 10 | 10 | 2.17 |
| 100 | 100 | 2.46 (ceiling: 2.5) |

The *first* mention tells you most of what you need: the document is about this word. Extra mentions add a little confidence, not a lot.

### Fix 2: Length fairness (knob `b`)
A long document contains more words, so it matches more by chance. BM25 compares each document's length to the **average** length:

| Doc length (avg = 30) | Weight of one mention |
|---|---|
| 15 (short) | 1.29 |
| 30 (average) | 1.00 |
| 60 (long) | 0.69 |

`b = 0` turns this off, and `b = 1` applies it fully. 0.75 is the standard default.

### The formula
$$
\text{score}(q, d) = \sum_{w \in q} \text{idf}(w) \cdot \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

- **f**: times word *w* appears in document *d* (a raw count, not divided by length)
- **|d|**: length of *d*; **avgdl**: average document length
- **k1 ≈ 1.2–2.0**: saturation; **b ≈ 0.75**: length normalization

Because `f` sits on both the top and the bottom of the fraction, the result flattens as `f` grows. That's the saturation.

> Real BM25 libraries use a slightly smoothed idf, `log((N − df + 0.5) / (df + 0.5) + 1)`, which never goes negative. We reuse our TF-IDF `idf`; the idea is the same.

### The spam test
A file containing only `python python python python python python python python` is added to the 4 practice files. Query: **"python snake"**.

| Method | 1st | 2nd |
|---|---|---|
| Keyword | python_snake.txt (2) | python_language / spam (1) |
| TF-IDF | **spam.txt (0.511)** ❌ | python_snake.txt (0.094) |
| BM25 | **python_snake.txt (2.245)** ✅ | spam.txt (1.170) |

TF-IDF ranks eight repeated words 5× above the real answer. BM25's saturation caps what repetition can earn.

### Limits
- **Still no meaning.** For "how long can a python sleep", all three methods rank documents by shared words like "can" and "sleep". None knows the question is about a snake. → Semantic search.
- **Two knobs to tune.** The defaults (k1 = 1.5, b = 0.75) work well almost everywhere.

Code: `term_score()`, `score()` and `search()` in `src/ragi/retrieval/bm25.py`. Compare all methods: `uv run python -m ragi.retrieval.compare`

---

## 6. Semantic search

Every method so far compares **words**. Semantic search compares **meaning**.

### Embeddings
An **embedding model** is a neural network trained on huge amounts of text to turn any text into a fixed-length list of numbers, a **vector**, so that texts with similar meaning get similar vectors.

```
"cat" → [0.037, 0.051, -0.000, 0.060, -0.117, ...]   ← 384 numbers (all-MiniLM-L6-v2)
```

No single number means "animal" or "furry". Meaning is spread across all of them. What matters is how vectors relate **to each other**. (Dimensions are covered properly in Phase 2.)

### Cosine similarity
Imagine vectors as arrows. Two texts are similar when their arrows **point in the same direction**:

```
cat    = [0.9, 0.1]  ↗
kitten = [0.8, 0.2]  ↗   nearly the same direction → similar
coffee = [0.1, 0.9]  ↑   different direction       → not similar
```

$$
\cos(A, B) = \frac{A \cdot B}{\|A\| \, \|B\|} = \frac{\sum_i A_i B_i}{\sqrt{\sum_i A_i^2} \cdot \sqrt{\sum_i B_i^2}}
$$

- **Top (dot product):** multiply matching numbers and add them up. It's big when the arrows agree.
- **Bottom (lengths):** divides out the arrows' lengths, so only the *direction* counts.
- **Result:** 1 = same direction (same meaning), 0 = unrelated, −1 = opposite.

Worked by hand with cat and kitten: dot = 0.9·0.8 + 0.1·0.2 = 0.74, lengths = 0.906 and 0.825, so cos = 0.74 / (0.906 · 0.825) = **0.99**. Cat and coffee give **0.22**.

### Real similarities (all-MiniLM-L6-v2)
| A | B | cosine |
|---|---|---|
| cat | kitten | **0.788** |
| cat | feline | 0.698 |
| cat | coffee | 0.429 |
| python snake | a large reptile | 0.439 |
| python snake | programming language | 0.261 |

### Semantic search
1. Embed every document (the 4 files → a 4 × 384 table).
2. Embed the question (→ 384 numbers).
3. Cosine similarity between the question and each document.
4. Sort, highest first, and keep the top k.

### Where it beats every word-based method
| Question | Keyword / TF-IDF / BM25 | Semantic |
|---|---|---|
| `cat` | nothing (the word isn't in any file) | **felines.txt** (0.459) |
| `what helps me stay awake` | nothing | **coffee.txt** (0.356) |
| `a reptile that squeezes its prey` | keyword ties on "a"; TF-IDF/BM25 only match "prey" | **python_snake.txt** (0.398) |

### Limits
- **It always returns something.** Nonsense like `asdfgh` still returns 3 documents (with low scores). Word-based methods would correctly return nothing. RAG has to handle "no good answer", for example with a minimum score.
- **Exact terms can get fuzzy.** Product codes, names and rare words (e.g. "RFC-7231") are matched more reliably by BM25. → That's why real systems use **hybrid search** (BM25 + semantic), in Phase 2.
- **It costs compute.** Every document must be embedded. Our `search` re-embeds them on every question, which is fine for 4 files and impossible for 10,000. → **Vector databases** store embeddings once (Phase 2).

Code: `embed()`, `similarity()` and `search()` in `src/ragi/retrieval/semantic.py`

---

## 7. Simple RAG

Everything so far was the **R**. Now the **A** and the **G**.

### Talking to an LLM
LLM APIs work like a chat: you send a list of **messages**, each with a **role**, and get text back.

| Role | Purpose | In our RAG |
|---|---|---|
| `system` | the rules the model must follow | "answer ONLY from the context…" |
| `user` | the request | the context + the question |
| `assistant` | the model's reply | the answer |

Most providers (OpenAI, Groq, OpenRouter, Ollama, …) accept the same **OpenAI-compatible** format. Changing provider means changing three settings: API key, base URL, model name.

### The three steps
```
1. Retrieve   semantic.search(question, docs, k=2)  →  top 2 documents
2. Augment    prompt = "Context:\n[felines.txt]\n<text>\n\n[coffee.txt]\n<text>\n\nQuestion: ..."
3. Generate   llm.ask(prompt, system=RULES)        →  answer with [source]
```

### The rules (system prompt)
```
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".
Cite the source file name in square brackets, like [coffee.txt].
```
Each line has a job:
- **ONLY the context**: use *your* data, not the model's memory.
- **"I don't know"**: an honest refusal instead of a confident guess (hallucination).
- **Cite the source**: a user can check the answer.

### Real results
| Question | Without RAG (LLM memory) | With RAG |
|---|---|---|
| How long can a kitten sleep? | "12–16 hours a day…" | "up to twenty hours a day. **[felines.txt]**" |
| Who created Python? | (knows it anyway) | "Guido van Rossum. **[python_language.txt]**" |
| Who invented coffee? | (would guess a story) | "**I don't know**" |

The first row is the whole point: the LLM's general knowledge says 12–16 hours, but *your* document says 20, and RAG answers from your document.

Notice the third answer: *"I don't know [coffee.txt]"*. The model cited a file even while saying it doesn't know. LLMs follow rules *mostly*, not perfectly. Measuring how often they follow them is what **evaluation** (Phase 2) is for.

### Limits of this simple version
| Limit | Fixed in |
|---|---|
| Whole files go into the prompt. A 100-page PDF wouldn't fit | Chunking (Phase 2) |
| Every question re-embeds every document | Vector DB / ChromaDB (Phase 2) |
| Semantic search always returns k results, even irrelevant ones | Minimum score / hybrid search (Phase 2) |
| The same question twice costs two LLM calls | Redis cache (Phase 2) |
| "Does it work?" is judged by eye | Evaluation (Phase 2) |

Code: `ask()` in `src/ragi/llm.py`; `RULES`, `build_context()` and `answer()` in `src/ragi/rag.py`. Run it with `uv run python -m ragi.rag`.

---

## Glossary
| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: find relevant text, then have an LLM answer from it |
| **LLM** | Large Language Model, e.g. GPT, Claude, Llama |
| **Hallucination** | An LLM stating something false with confidence |
| **Document** | One loaded file: its text plus its source |
| **Token** | One unit of text after splitting, for us a lowercase word |
| **Tokenizing** | Splitting text into tokens |
| **Regex** | Regular expression, a pattern language for finding text |
| **Corpus / collection** | All the documents you search over |
| **Top-k** | The best `k` results of a search |
| **TF** | Term Frequency: how often a word appears in one document, relative to its length |
| **DF** | Document Frequency: how many documents contain a word |
| **IDF** | Inverse Document Frequency: log(N / df), high for rare words |
| **TF-IDF** | tf × idf, a word's weight in a document |
| **BM25** | TF-IDF plus saturation (k1) and length normalization (b); the standard keyword ranking |
| **Saturation** | each repeat of a word adds less than the previous one |
| **avgdl** | average document length in the collection |
| **Embedding / vector** | a fixed-length list of numbers representing a text's meaning |
| **Embedding model** | a neural network that produces embeddings, e.g. all-MiniLM-L6-v2 |
| **Cosine similarity** | how closely two vectors point in the same direction (−1 to 1) |
| **Tensor** | PyTorch's name for an array of numbers in any number of dimensions |
| **Semantic search** | search by meaning, using embeddings |
| **System prompt** | the rules message sent before the user's message |
| **Context** | the retrieved text pasted into the prompt |
| **OpenAI-compatible API** | the common chat format most LLM providers accept |
| **Mocking** | replacing a real dependency (like an API client) with a fake one in tests |
