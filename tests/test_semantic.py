from pytest import approx

from ragi.loaders import Document, load_folder
from ragi.retrieval.semantic import embed, similarity, search


def test_embed_returns_384_numbers():
    assert embed("cat").shape == (384,)


def test_text_is_identical_to_itself():
    assert similarity("cat", "cat") == approx(1.0, abs=0.001)


def test_similar_meaning_scores_higher():
    assert similarity("cat", "kitten") > similarity("cat", "coffee")


def test_search_finds_meaning_without_shared_words():
    results = search("cat", load_folder())
    assert results[0][1].source == "felines.txt"


def test_search_always_returns_k_results():
    docs = [Document(text=t, source=f"{i}.txt") for i, t in enumerate(["coffee", "tea", "snake", "kitten"])]
    assert len(search("asdfgh", docs, k=3)) == 3
