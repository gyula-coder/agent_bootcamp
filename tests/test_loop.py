from agent import config
from agent import loop


def test_call_model_uses_configured_openai_compatible_provider(monkeypatch):
    created_clients = []
    completion_calls = []

    class FakeMessage:
        def model_dump(self, exclude_none: bool = True):
            return {"role": "assistant", "content": "provider reply"}

    class FakeCompletions:
        def create(self, **kwargs):
            completion_calls.append(kwargs)
            return type("Response", (), {"choices": [type("Choice", (), {"message": FakeMessage()})]})

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str):
            created_clients.append({"api_key": api_key, "base_url": base_url})
            self.chat = FakeChat()

    monkeypatch.setattr(config, "BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(config, "MODEL", "provider-model")
    monkeypatch.setattr(config, "API_KEY", "provider-key")
    monkeypatch.setattr(loop, "OpenAI", FakeOpenAI)

    result = loop.call_model([{"role": "user", "content": "hello"}])

    assert result == {"role": "assistant", "content": "provider reply"}
    assert created_clients == [
        {"api_key": "provider-key", "base_url": "https://provider.example/v1"}
    ]
    assert completion_calls[0]["model"] == "provider-model"
