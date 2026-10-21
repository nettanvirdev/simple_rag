from sentence_transformers import SentenceTransformer, util
from ragi import config
from ragi.loaders import Document

model = SentenceTransformer(config.EMBEDDING_MODEL)


def embed(text: str):
    return model.encode(text)


def similarity(a: str, b: str) -> float:
    similarity = util.cos_sim(embed(a), embed(b)).item()
    return similarity


def search(query: str, docs: list[Document], k: int = 3) -> list[tuple[float, Document]]:
    # ALL docs at once → table of shape (4, 384), one row per doc
    doc_vectors = model.encode([d.text for d in docs])
    # the question → (384,)
    query_vector = model.encode(query)
    scores = util.cos_sim(query_vector, doc_vectors)[0].tolist()
    results = []
    # ponytail: re-embeds every doc on every search; a vector DB stores them once (section 10)
    for s, doc in zip(scores, docs):
        results.append((s, doc))

    return sorted(results, key=lambda s: s[0], reverse=True)[:k]


# only runs when you start this file directly, not when another file imports it
if __name__ == "__main__":
    from ragi.loaders import load_folder
    docs = load_folder()
    while True:                           # loop forever
        query = input("search> ")         # you type a question
        for s, doc in search(query, docs):
            print(s, doc.source)
