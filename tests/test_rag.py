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
