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
