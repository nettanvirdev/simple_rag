---
title: "Step 12: Your first RAG"
series: "Simple RAG"
step: 12
slug: "step-12-your-first-rag"
date: "2026-10-21"
tag: "RAG"
excerpt: "Retrieve the best documents, paste them into the prompt, and get an LLM answer from your own data, with citations and an honest \"I don't know\"."
minutes: 12
---

# Step 12: Your first RAG

**You'll build:** `src/ragi/rag.py` with `build_context()` and `answer()`, plus an interactive `ask>` box.
**You'll learn:** retrieve → augment → generate, system-prompt rules, `"\n\n".join`, and testing a whole pipeline with a fake LLM.

---

## 1. Theory

**R**etrieve: find the top k documents with semantic search (Step 10). **A**ugment: paste them into the prompt as "context". **G**enerate: the LLM answers using **only** that context, says "I don't know" otherwise, and cites the file. The rules in the system message are what make the answer trustworthy.

```
question ──► semantic.search(question, docs, k=2) ──► top 2 documents
                                                          │
                                                          ▼
     answer ◄── ask(prompt, system=RULES) ◄── prompt = context + question
```

More theory: [concepts.md, Simple RAG](../concepts.md#7-simple-rag).

---

## Before vs after (measured)

| | Before (bare LLM) | After (RAG) |
|---|---|---|
| "How long can a kitten sleep?" | "12–16 hours a day" (model memory) | "up to twenty hours a day. [felines.txt]" |
| "Who invented coffee?" (not in the files) | a guess | "I don't know" |
| Source shown | no | yes, the file name |

---

## 2. Code example

### 2.1 The rules (given)
```python
RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know\". "
    "Cite the source file name in square brackets, like [coffee.txt]."
)
```
Strings written next to each other inside `( )` join into one long string. `\"` puts a quote mark *inside* a string.

Each rule has a job:

| Rule | Why |
|---|---|
| ONLY the context | use *your* data, not the model's memory |
| say "I don't know" | an honest refusal instead of a confident guess (hallucination) |
| cite the source | the user can check the answer |

### 2.2 New idea: `join`
`join` glues the items of a list together, with a separator between them:
```python
"\n\n".join(["first", "second"])    # 'first\n\nsecond' ← \n = new line, so this puts a blank line between them
```

---

## 3. Goal

Create `src/ragi/rag.py`. Imports: `Document` from `ragi.loaders`, `semantic` from `ragi.retrieval`, and **`ask` from `ragi.llm`**. Add `RULES` (above) and two functions:

**`build_context(results: list[tuple[float, Document]]) -> str`**
1. `parts = []`
2. For each `s, doc` in `results`, append `f"[{doc.source}]\n{doc.text}"`: the file name as a label, then its text.
3. Return `"\n\n".join(parts)`.

**`answer(question: str, docs: list[Document], k: int = 2) -> str`**
1. `results = semantic.search(question, docs, k)`
2. `context = build_context(results)`
3. `prompt = f"Context:\n{context}\n\nQuestion: {question}"`
4. Return `ask(prompt, system=RULES)`.

Add the box at the bottom (given):
```python
if __name__ == "__main__":
    from ragi.loaders import load_folder
    docs = load_folder()
    while True:
        question = input("ask> ")
        print(answer(question, docs))
```

### ✅ Checkpoint
```bash
uv run python -m ragi.rag
```

| Ask | Answer |
|---|---|
| `How long can a kitten sleep?` | A kitten can sleep up to twenty hours a day. [felines.txt] |
| `Who created Python?` | Guido van Rossum. [python_language.txt] |
| `Who invented coffee?` | I don't know [coffee.txt]. |

The first row is the whole point: in Step 11, the bare LLM said **12–16 hours** from its own memory. With RAG, it answers **twenty hours**, from *your* document, and cites it.

Notice the third answer: the model cited a file **while saying it doesn't know**. LLMs follow rules *mostly*, not perfectly. Measuring how often they follow them is what **evaluation** is for.

---

## 4. Test

All three use the `fake_llm` fixture from Step 11, so there are no API calls.

**This test checks:** each document in the context is labelled with its file name. **How:** build a context from two documents and look for `[file]\ntext` for each one.

**This test checks:** RAG's rules really reach the LLM. **How:** the first message the fake receives must be the system message with exactly `RULES`.

**This test checks:** retrieval puts the right document into the prompt. **How:** for the kitten question, the prompt must contain `[felines.txt]`, its text ("twenty hours"), and the question itself.

```python
# tests/test_rag.py
from ragi.loaders import Document, load_folder
from ragi.rag import RULES, build_context, answer


def test_build_context_labels_each_document():
    results = [
        (0.9, Document(text="kittens sleep a lot", source="felines.txt")),
        (0.5, Document(text="coffee keeps you awake", source="coffee.txt")),
    ]
    context = build_context(results)
    assert "[felines.txt]\nkittens sleep a lot" in context
    assert "[coffee.txt]\ncoffee keeps you awake" in context


def test_answer_sends_rules_as_system_message(fake_llm):
    answer("How long can a kitten sleep?", load_folder())
    assert fake_llm[0] == {"role": "system", "content": RULES}


def test_answer_puts_retrieved_document_in_prompt(fake_llm):
    answer("How long can a kitten sleep?", load_folder())
    prompt = fake_llm[1]["content"]
    assert "[felines.txt]" in prompt
    assert "twenty hours" in prompt
    assert "Question: How long can a kitten sleep?" in prompt
```

```bash
uv run pytest -v
```

---

## 5. Make it a module

`answer()` is the heart of the app. A production RAG keeps the same idea and upgrades each part:

| Limit of this simple version | Production upgrade |
|---|---|
| Whole files go into the prompt; a 100-page PDF wouldn't fit | **Chunking**: split documents into small pieces |
| Every question re-embeds every document | **Vector database** (e.g. ChromaDB): embed once, store, reuse |
| Semantic search always returns k results, even irrelevant ones | **Minimum score** and **hybrid search** (BM25 + semantic) |
| The same question twice costs two LLM calls | **Cache** (e.g. Redis) |
| "Does it work?" is judged by eye | **Evaluation**: test questions with expected answers, scored automatically |
| Only usable from the terminal | **API** (e.g. FastAPI) so a website or app can call it |

---

<details>
<summary><strong>Show solution: <code>src/ragi/rag.py</code></strong></summary>

```python
from ragi.loaders import Document
from ragi.retrieval import semantic
from ragi.llm import ask


RULES = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know\". "
    "Cite the source file name in square brackets, like [coffee.txt]."
)


def build_context(results: list[tuple[float, Document]]) -> str:
    parts = []

    for s, doc in results:
        parts.append(f"[{doc.source}]\n{doc.text}")
    return "\n\n".join(parts)


def answer(question: str, docs: list[Document], k: int = 2) -> str:
    results = semantic.search(question, docs, k)
    context = build_context(results)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return ask(prompt, system=RULES)


if __name__ == "__main__":
    from ragi.loaders import load_folder
    docs = load_folder()
    while True:
        question = input("ask> ")
        print(answer(question, docs))
```

</details>

---

## 🎉 You built a RAG

In 12 steps, from an empty folder:

| Steps | What you built |
|---|---|
| 1–2 | A production-style project and a loader that turns files into `Document`s |
| 3–4 | **Keyword search**: count shared words |
| 5–6 | **TF-IDF**: rare words matter more, common words count for nothing |
| 7–8 | **BM25**: repetition stops paying off, long documents get no free advantage |
| 9–10 | **Semantic search**: match by meaning with embeddings |
| 11–12 | An LLM client and a **RAG** that answers from your documents, cites them, and says "I don't know" |

Every method fixes the previous one's weakness:
```
keyword  →  TF-IDF  →  BM25  →  semantic search  →  RAG
```

Review the whole series in the [overview](README.md), and the theory and math in [RAG Concepts](../concepts.md).

---

**Previous:** [← Step 11: Talking to an LLM](step-11-talking-to-an-llm.md)
