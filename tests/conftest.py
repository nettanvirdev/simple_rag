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
