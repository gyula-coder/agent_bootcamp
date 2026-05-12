import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any


_MOCK_COLLECTIONS = [
    {
        "name": "agent_bootcamp",
        "description": "Learning notes and PRD content for the Agent Bootcamp project.",
    },
    {
        "name": "rag_bootcamp",
        "description": "Mock collection for earlier RAG learning materials.",
    },
]

_MOCK_CHUNKS = [
    {
        "collection": "agent_bootcamp",
        "chunk_id": "agent-prd-001",
        "source": "docs/agent-7-day-learning-prd.md",
        "content": "Day 1 focuses on a handwritten tool calling loop with calculator, time, retrieval, and collection listing tools.",
        "score": 0.92,
    },
    {
        "collection": "agent_bootcamp",
        "chunk_id": "agent-prd-002",
        "source": "docs/agent-7-day-learning-prd.md",
        "content": "The recommended project structure separates api, agent, tools, memory, eval, and docs modules.",
        "score": 0.88,
    },
    {
        "collection": "rag_bootcamp",
        "chunk_id": "rag-note-001",
        "source": "mock/rag-notes.md",
        "content": "RAG retrieval should return chunk content, source, chunk id, and ranking or score information.",
        "score": 0.81,
    },
]


def retrieve_knowledge(
    query: str,
    top_k: int = 3,
    strategy: str = "adaptive",
) -> list[dict[str, Any]]:
    if not query.strip():
        return {"ok": False, "error": "query must not be empty"}

    top_k = min(max(top_k, 1), 10)
    payload = {
        "query": query.strip(),
        "top_k": top_k,
        "chunking_strategy": strategy,
        "threshold": 0.6,
    }
    body = json.dumps(payload).encode("utf-8")
    retrieve_url = "http://127.0.0.1:8000/retrieve/rerank"
    req = Request(
        url = retrieve_url,
        data = body,
        headers = {"Content-Type": "application/json"},
        method = "POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = (
            exc.read().decode("utf-8", errors="ignore")
            if hasattr(exc, "read")
            else str(exc)
        )
        raise RuntimeError(f"{retrieve_url} HTTP错误: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{retrieve_url} 网络错误: {exc.reason}") from exc

    hits = (((response or {}).get("data") or {}).get("hits") or [])
    normalized_hits: list[dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        chunk_id = str(hit.get("chunk_id", "")).strip()
        chunk_index_raw = hit.get("chunk_index", -1)
        score_raw = hit.get("rerank_score", hit.get("rrf_score", hit.get("score", 0.0)))

        try:
            chunk_index = int(chunk_index_raw)
        except (TypeError, ValueError):
            chunk_index = -1

        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0

        normalized_hits.append(
            {
                "chunk_id": chunk_id,
                "content": str(hit.get("content", "")),
                "source": str(hit.get("source", "")),
                "chunk_index": chunk_index,
                "score": score,
            }
        )
    return normalized_hits


def list_collections() -> dict[str, Any]:
    return {"ok": True, "collections": _MOCK_COLLECTIONS}
