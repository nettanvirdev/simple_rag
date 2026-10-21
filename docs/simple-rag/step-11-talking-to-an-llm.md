---
title: "Step 11: Talking to an LLM"
series: "Simple RAG"
step: 11
slug: "step-11-talking-to-an-llm"
date: "2026-10-21"
tag: "RAG"
excerpt: "Send messages to any OpenAI-compatible LLM from Python, and test it with a fake client so your tests never touch the network."
minutes: 10
---

# Step 11: Talking to an LLM

**You'll build:** `src/ragi/llm.py` with one function, `ask(prompt, system="")`.
**You'll learn:** chat roles, OpenAI-compatible APIs, dictionaries, and **mocking** (testing with a fake client).

---

## 1. Theory

An LLM API works like a chat: you send a list of **messages** and get text back. Each message has a **role**. Most providers speak the same "OpenAI-compatible" format, so switching provider only means changing three values in `.env`: API key, base URL, model.

| Role | Purpose | In our RAG (Step 12) |
|---|---|---|
| `system` | the rules the model must follow | "answer ONLY from the context…" |
| `user` | the request | the context + the question |
| `assistant` | the model's reply | the answer |

Providers that accept this format include OpenAI, Groq, OpenRouter, Together, and a local **Ollama** (`http://localhost:11434/v1`, free and offline).

---

## Before vs after (measured)

| | Before | After (`llm.ask`) |
|---|---|---|
| Output | ranked files only | a written answer |
| "How long can a kitten sleep?" | — | "12–16 hours a day" (from the model's memory) |
| What `felines.txt` says | — | "up to twenty hours a day" (not used yet) |

---

## 2. Code example

### 2.1 Install and configure
```bash
uv add openai
```
Put **your own** provider's values in `.env`. They come from your provider's dashboard; never commit them:
```bash
LLM_API_KEY=your-key-here
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
```

### 2.2 The client (given)
```python
from openai import OpenAI          # the client library; works with ANY OpenAI-compatible server, not just OpenAI
from ragi import config

client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)   # the connection, made once

response = client.chat.completions.create(
    model=config.LLM_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},   # the rules
        {"role": "user", "content": "Say hello"},                        # the question
    ],
)
response.choices[0].message.content    # the reply text
```

### 2.3 New idea: dictionaries `{ }`
A dictionary stores labelled values, **key → value**:
```python
msg = {"role": "user", "content": "Say hello"}
msg["role"]       # 'user'
msg["content"]    # 'Say hello'
```
So `messages` is a **list of dictionaries**, one per message.

**Why `choices[0]`?** An API can return several alternative replies. We ask for one and take the first.

---

## 3. Goal

Create `src/ragi/llm.py` with `client` (from 2.2) and one function:
```python
def ask(prompt: str, system: str = "") -> str | None:
```
`str | None` means "a string, or `None`". The library may return no text in rare cases.

Steps:
1. `messages = []`
2. If `system` isn't empty, append `{"role": "system", "content": system}`, using **the argument** `system`, not a fixed text.
3. Append `{"role": "user", "content": prompt}`.
4. Call `client.chat.completions.create(...)` with `config.LLM_MODEL` and your `messages`.
5. Return `response.choices[0].message.content`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.llm import ask; print(ask('Reply with exactly: hello'))"
```
Expected: `hello`

### 🧪 Experiment: why we need RAG
```bash
uv run python -c "from ragi.llm import ask; print(ask('How long can a kitten sleep? One sentence.'))"
```
A test run gave: *"Kittens typically sleep 12–16 hours a day…"*. But our `felines.txt` says **up to twenty hours**. The LLM answers from its own memory, not from your documents. Step 12 fixes that.

---

## 4. Test

Tests should **not** call the real API: it's slow, costs money, and fails without internet. So we replace the real `client` with a **fake one** during the test. This is called **mocking**. The fake records the messages it receives and always replies `"fake reply"`.

The fake lives in `tests/conftest.py` as a **fixture**. pytest automatically shares anything in `conftest.py` with every test file, so any test can use the fake by naming `fake_llm` as a parameter, exactly like `tmp_path` in Step 2.

```python
# tests/conftest.py
from types import SimpleNamespace

import pytest

from ragi import llm


@pytest.fixture
def fake_llm(monkeypatch):
    """Swap the real API client for a fake one. Gives the test the list of messages the fake receives."""
    sent = []

    def fake_create(model, messages):
        sent.extend(messages)
        # shaped like a real response: response.choices[0].message.content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fake reply"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(llm, "client", fake_client)  # undone automatically after the test
    return sent
```

| Piece | Meaning |
|---|---|
| `monkeypatch.setattr(llm, "client", fake_client)` | during this test, `llm.client` **is** the fake; pytest puts the real one back afterwards |
| `SimpleNamespace(a=1)` | a quick object where `.a` gives `1`, used to build something shaped like `response.choices[0].message.content` |
| `sent` | the list of messages your `ask` sent, which the test can inspect |

**This test checks:** `ask` returns the reply text. **How:** the fake always answers `"fake reply"`, so that's what `ask` must return.

**This test checks:** without a `system` argument, only the user message is sent. **How:** compare the full list of sent messages.

**This test checks:** the caller's rules really reach the LLM. **How:** pass a `system` text and check that the first sent message contains **exactly** that text.

```python
# tests/test_llm.py
from ragi import llm


def test_ask_returns_reply_text(fake_llm):
    assert llm.ask("hi") == "fake reply"


def test_ask_without_system_sends_only_user_message(fake_llm):
    llm.ask("hi")
    assert fake_llm == [{"role": "user", "content": "hi"}]


def test_ask_sends_the_given_system_rules(fake_llm):
    llm.ask("hi", system="Only use the context")
    assert fake_llm[0] == {"role": "system", "content": "Only use the context"}
```

```bash
uv run pytest -v
```
These run instantly, for free, and without internet.

---

## 5. Make it a module

In Step 12, `rag.py` calls `ask(prompt, system=RULES)`. The rules ("only use the context, otherwise say I don't know, cite the file") travel through the `system` argument. That's why the third test matters so much: if `ask` ignored `system`, RAG's rules would silently disappear.

---

<details>
<summary><strong>Show solution: <code>src/ragi/llm.py</code></strong></summary>

```python
from openai import OpenAI
from ragi import config

client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def ask(prompt: str, system: str = "") -> str | None:
    messages = []
    if system != "":
        messages.append(
            {"role": "system", "content": system})

    messages.append(
        {"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages
    )

    return response.choices[0].message.content
```

</details>

---

**Previous:** [← Step 10: Semantic search](step-10-semantic-search.md)

**Next:** [Step 12: Your first RAG →](step-12-your-first-rag.md)
