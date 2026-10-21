from ragi.loaders import load_folder
# import whole files (modules), not single functions
from ragi.retrieval import keyword, tfidf, bm25, semantic

docs = load_folder()
while True:
    query = input("compare> ")
    for name, method in [("keyword", keyword), ("tfidf", tfidf), ("bm25", bm25), ("semantic", semantic)]:
        print(f"--- {name}")
        for s, doc in method.search(query, docs):    # each module's own search
            print(f"  {s:.3f}  {doc.source}")        # :.3f = show 3 decimals
