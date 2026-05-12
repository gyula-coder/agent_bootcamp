import json

from tools import rag


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "data": {
                    "hits": [
                        {
                            "chunk_id": "chunk-1",
                            "content": "RAG should cite sources.",
                            "source": "docs/rag.md",
                            "chunk_index": 2,
                            "rerank_score": 0.91,
                        }
                    ]
                }
            }
        ).encode("utf-8")


def test_retrieve_knowledge_sends_top_k_and_adaptive_strategy(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        assert timeout == 30
        return FakeResponse()

    monkeypatch.setattr(rag, "urlopen", fake_urlopen)

    result = rag.retrieve_knowledge("什么是 RAG？", top_k=3)

    payload = json.loads(requests[0].data.decode("utf-8"))
    assert payload["top_k"] == 3
    assert payload["chunking_strategy"] == "adaptive"
    assert result == [
        {
            "chunk_id": "chunk-1",
            "content": "RAG should cite sources.",
            "source": "docs/rag.md",
            "chunk_index": 2,
            "score": 0.91,
        }
    ]
