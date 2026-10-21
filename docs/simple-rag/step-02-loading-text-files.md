---
title: "Step 2: Loading text files"
series: "Simple RAG"
step: 2
slug: "step-02-loading-text-files"
date: "2026-10-21"
tag: "RAG"
excerpt: "Turn every .txt file in a folder into a Document with its text and source, the one shape every later step works with."
minutes: 12
---

# Step 2: Loading text files

**You'll build:** `src/ragi/loaders.py`, which turns files into `Document(text, source)`.
**You'll learn:** dataclasses, type hints, generators (`glob`), `sorted`, `continue`, list comprehensions, and pytest's `tmp_path`.

---

## 1. Theory

The first step of RAG is turning files into one shape: **the text plus where it came from**. Everything after loading (search, chunking, citations) only ever sees this shape, never files. So adding PDFs later means writing one new loader, and nothing downstream changes. More in [concepts: Documents and loading](../concepts.md#1-documents-and-loading).

Practice data: put the four `.txt` files from the repo's `data/` folder into your own `data/` folder.

---

## Before vs after (measured)

| | Before | After |
|---|---|---|
| Documents loaded from `data/` | 0 | 4 (`coffee`, `felines`, `python_language`, `python_snake`) |
| Order | depends on the computer | sorted by file name |
| Empty files | would become empty documents | skipped |

---

## 2. Code example

### 2.1 A record type and one file loader
Two new ideas: a **dataclass** (a simple record with named fields) and **type hints** (labels that say what type a value is).

```python
from dataclasses import dataclass   # tool for making simple "record" types
from pathlib import Path

@dataclass                          # writes the setup code for us, so Document(text=..., source=...) just works
class Document:                     # a new type, like str or int, but ours
    text: str                       # field 1: the file's content
    source: str                     # field 2: the file name, used later to cite where an answer came from

def load_txt(path: Path) -> Document:             # hints: takes a Path, returns a Document (Python doesn't enforce them)
    text = path.read_text(encoding="utf-8")       # read the whole file into one string; utf-8 keeps non-English text safe
    return Document(text=text, source=path.name)  # .name = just "coffee.txt", not the full path
```

```python
doc = load_txt(Path("data/coffee.txt"))
doc.text     # "Coffee is brewed from..."
doc.source   # "coffee.txt"
```

### 2.2 Getting every file path from a folder
`folder.glob("*.txt")` finds every file whose name ends in `.txt`. But printing it is confusing:

```python
print(folder.glob("*.txt"))
# <generator object Path.glob at 0x000001DC...>   ← not the paths!

print(list(folder.glob("*.txt")))
# [WindowsPath('data/coffee.txt'), WindowsPath('data/felines.txt'),
#  WindowsPath('data/python_language.txt'), WindowsPath('data/python_snake.txt')]
```

A **generator** is like a ticket machine: it doesn't hold all the tickets, it prints the next one each time you ask. `print` only shows you the machine. You get the actual paths in two ways:
- **`list(...)` or `sorted(...)`** asks for every path at once and gives a normal list.
- **a `for` loop** asks for one path per round:
  ```python
  for path in folder.glob("*.txt"):   # data/coffee.txt, then data/felines.txt, ...
      print(path.name)
  ```

Try it in the Python playground (`uv run python`, then `exit()` to leave):
```python
from pathlib import Path
folder = Path("data")
folder.glob("*.txt")
list(folder.glob("*.txt"))
sorted(folder.glob("*.txt"))
```

---

## 3. Goal

Create `src/ragi/loaders.py` with `Document` and `load_txt` from above (type them, don't paste), plus:

```python
def load_folder(folder: Path = config.DATA_DIR) -> list[Document]:
```
`= config.DATA_DIR` is a **default argument**: `load_folder()` uses `data/`, and `load_folder(other_path)` overrides it.

Steps, one line of Python each:
1. Make an empty list called `docs`.
2. Loop over `sorted(folder.glob("*.txt"))`. Sorting means the same order on every computer.
3. Inside the loop, `doc = load_txt(path)`.
4. If `doc.text.strip()` is empty, skip it with `continue` (jump to the next round). `"   ".strip()` is `""`.
5. Otherwise, `docs.append(doc)`.
6. After the loop, `return docs`.

### ✅ Checkpoint
```bash
uv run python -c "from ragi.loaders import load_folder; docs = load_folder(); print(len(docs)); print(docs[0])"
```
Expected: `4`, then `Document(text='Coffee is brewed from roasted beans.\n...', source='coffee.txt')`. Dataclasses know how to print themselves.

---

## 4. Test

These tests never touch your real `data/` folder. pytest's built-in **`tmp_path`** gives each test a fresh, empty folder (a `Path`), so the test creates its own input files. That makes them test **code behavior**, not your machine.

**This test checks:** a file's text is read correctly. **How:** we write "hello rag" ourselves, then load it.

```python
from ragi.loaders import load_txt, load_folder


def test_load_txt_reads_text(tmp_path):            # pytest sees the name tmp_path and passes in a new temp folder
    file = tmp_path / "a.txt"
    file.write_text("hello rag", encoding="utf-8")
    doc = load_txt(file)
    assert doc.text == "hello rag"
```

**This test checks:** non-`.txt` files are ignored. **How:** the folder has `a.txt` (should load) and `notes.md` (shouldn't). Without `a.txt`, the list would always be empty and the test would pass even with broken code.

```python
def test_load_folder_ignores_non_txt(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("hello", encoding="utf-8")
    file2 = tmp_path / "notes.md"
    file2.write_text("test", encoding="utf-8")
    docs = load_folder(tmp_path)

    sources = []
    for doc in docs:
        sources.append(doc.source)
    assert "notes.md" not in sources

    # the same, as a one-line list comprehension
    sources2 = [doc.source for doc in docs]
    assert "notes.md" not in sources2
```

`docs` holds **Document objects**, not strings, so `"notes.md" not in docs` would compare a string with Documents and always be True: a test that can never fail. Pull out the names first. `[doc.source for doc in docs]` reads "doc.source, for each doc in docs" and builds the same list as the loop.

**This test checks:** empty files are skipped. **How:** the only file is empty, so nothing should load.

```python
def test_load_folder_skips_empty_files(tmp_path):
    file = tmp_path / "empty.txt"
    file.write_text("", encoding="utf-8")
    docs = load_folder(tmp_path)
    assert len(docs) == 0
```

**This test checks:** files come out in alphabetical order. **How:** we create `b.txt` *first*, so only real sorting can put `a.txt` at position 0.

```python
def test_load_folder_is_sorted(tmp_path):
    file = tmp_path / "b.txt"
    file.write_text("test", encoding="utf-8")
    file2 = tmp_path / "a.txt"
    file2.write_text("test", encoding="utf-8")
    docs = load_folder(tmp_path)
    assert docs[0].source == "a.txt"      # docs[0] = first item, .source = its file name
```

```bash
uv run pytest -v
```

**Prove the tests can fail:** change `"*.txt"` to `"*"` in `loaders.py`. `test_load_folder_ignores_non_txt` should go red. Change it back.

---

## 5. Make it a module

Step 3 (keyword search) starts like this:
```python
from ragi.loaders import Document, load_folder
docs = load_folder()   # list[Document], and search only sees Documents
```
`Document(text, source)` is the contract between your modules. Keep the field names exactly `text` and `source`.

---

<details>
<summary><strong>Show solution: <code>src/ragi/loaders.py</code></strong></summary>

```python
from dataclasses import dataclass
from pathlib import Path
from ragi import config


@dataclass
class Document:
    text: str
    source: str


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(text=text, source=path.name)


def load_folder(folder: Path = config.DATA_DIR) -> list[Document]:

    docs = []
    for path in sorted(folder.glob("*.txt")):
        doc = load_txt(path=path)
        if doc.text.strip() == "":
            continue
        docs.append(doc)
    return docs
```

</details>

---

**Previous:** [← Step 1: Project setup](step-01-project-setup.md)

**Next:** [Step 3: Keyword search: tokenize and score →](step-03-keyword-tokenize-and-score.md)
