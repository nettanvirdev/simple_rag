---
title: "Step 1: Project setup"
series: "Simple RAG"
step: 1
slug: "step-01-project-setup"
date: "2026-10-21"
tag: "RAG"
excerpt: "Create a production-style Python project with uv, keep secrets in .env, and build the config module every other file imports."
minutes: 10
---

# Step 1: Project setup

**You'll build:** `pyproject.toml`, `.env`, and `src/ragi/config.py`, the one file that reads settings.
**You'll learn:** uv, the `src/` layout, environment variables, `pathlib`, and your first pytest test.

---

## 1. Theory

A production project keeps four kinds of things apart: **code** (`src/ragi/`), **input data** (`data/`), **generated data** (`storage/`), and **secrets** (`.env`). Tests live in `tests/` and check the code. **uv** manages the Python version, the libraries, and a private environment (`.venv/`) for this project.

The folders we grow into over the whole series:

```
ragi/
├── pyproject.toml        project info + dependencies (uv manages it)
├── uv.lock               exact installed versions (never edit by hand)
├── .env                  your real secrets (git-ignored)
├── .env.example          same keys, no secrets (committed)
├── data/                 input documents (.txt, .pdf)
├── storage/              generated data, rebuildable (git-ignored)
├── src/ragi/             the package: `from ragi import ...` works anywhere
│   ├── config.py         Step 1: all settings, read once from .env
│   ├── loaders.py        Step 2: files → documents
│   ├── retrieval/        Steps 3–10: four search methods
│   ├── llm.py            Step 11: talks to the LLM API
│   └── rag.py            Step 12: retrieve → prompt → answer
└── tests/                one test_<module>.py per module
```

**Why `src/ragi/`?** The folder with `__init__.py` is a *package*. uv installs it into `.venv`, so `from ragi import config` works from anywhere: tests, the terminal, and later a web API.

---

## Before vs after (measured)

| | Before | After |
|---|---|---|
| Project | empty folder | uv project with `src/` layout and `.env` |
| Places that read settings | none | 1 (`config.py`) |
| Tests | 0 | 8 passed |

---

## 2. Code example

### 2.1 Create the project
```bash
uv init --package --name ragi --python 3.12
uv add python-dotenv
uv add --dev pytest
```

| Command | What it does |
|---|---|
| `uv init --package` | creates `pyproject.toml` and `src/ragi/__init__.py` (the src layout) |
| `--python 3.12` | machine-learning libraries (torch, chromadb) support 3.12 most reliably |
| `uv add python-dotenv` | installs a library that reads `.env` files and records it in `pyproject.toml` |
| `uv add --dev pytest` | installs the test framework as a dev-only tool |

Run anything inside the project's environment with `uv run <command>`. You never need to "activate" anything.

### 2.2 Secrets: `.env.example` → `.env`
```bash
# .env.example (committed, no real values)
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DIR=storage/chroma
REDIS_URL=redis://localhost:6379/0
```
Copy it to `.env`, fill in real values, and add `.env` and `storage/` to `.gitignore`.

### 2.3 The config module
`config.py` is the **only** file that reads `.env`. Every other file asks `config` for settings.

```python
from pathlib import Path        # Path = a file/folder location; works the same on Windows and Linux
import os                       # os.getenv() reads environment variables
from dotenv import load_dotenv  # copies KEY=value lines from .env into the environment

ROOT = Path(__file__).resolve().parents[2]  # __file__ = this file. parents[0]=src/ragi, [1]=src, [2]=project root
load_dotenv(ROOT / ".env")                  # the / operator joins paths → <project>/.env

LLM_MODEL = os.getenv("LLM_MODEL", "")      # read a value; use "" if it's missing
```

---

## 3. Goal

Create `src/ragi/config.py` with these names:
- `ROOT` and `DATA_DIR`. `DATA_DIR` points to the `data` folder (hint: `ROOT / "data"`).
- `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- `EMBEDDING_MODEL`, `REDIS_URL`
- `CHROMA_DIR` as a **Path** under `ROOT`. Hint: `ROOT / os.getenv("CHROMA_DIR", "storage/chroma")`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi import config; print(config.DATA_DIR, config.CHROMA_DIR)"
```
Both should print **full paths**, ending in `data` and `storage/chroma`.

---

## 4. Test

pytest finds files named `test_*.py` and runs every function in them whose name starts with `test_`. `assert something` passes when `something` is True and fails when it's False. Add a message after a comma to explain a failure: `assert ok, "why it failed"`.

**This test checks:** the project root really is the project root. **How:** the root must contain `pyproject.toml`.

```python
# tests/test_config.py
from ragi import config

def test_root_has_pyproject():
    assert (config.ROOT / "pyproject.toml").exists()

def test_data_dir_exists():
    assert config.DATA_DIR.exists(), f"The data directory does not exist: {config.DATA_DIR}"
    assert config.DATA_DIR.name == "data"

def test_embedding_model_is_set():
    assert config.EMBEDDING_MODEL != ""
```

```bash
uv run pytest -v
```

**Try breaking it:** change `"data"` to `"dataa"` in `config.py`, rerun, read the red failure message, and change it back. A test you've never seen fail might not be testing anything.

> Tests that read `.env` values check **your machine**, not your code: they fail on a computer without your `.env`. From Step 2 on, tests check code behavior: given this input, expect that output.

---

## 5. Make it a module

From now on, every file gets its settings like this:
```python
from ragi import config
config.DATA_DIR   # never os.getenv(...) anywhere except config.py
```
Change a setting in one place, and every file sees it. Step 2's `loaders.py` uses `config.DATA_DIR` to find your text files.

---

<details>
<summary><strong>Show solution: <code>src/ragi/config.py</code></strong></summary>

```python
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

CHROMA_DIR = ROOT / os.getenv("CHROMA_DIR", "storage/chroma")
REDIS_URL = os.getenv("REDIS_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
```

</details>

---

**Next:** [Step 2: Loading text files →](step-02-loading-text-files.md)
