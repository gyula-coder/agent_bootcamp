from fastapi.testclient import TestClient

from api import main


def test_agent_chat_calls_agent_loop(monkeypatch):
    calls = []

    def fake_run_agent_loop(user_input: str, max_iterations: int = 5) -> str:
        calls.append((user_input, max_iterations))
        return "agent reply"

    monkeypatch.setattr(main, "run_agent_loop", fake_run_agent_loop)
    client = TestClient(main.app)

    response = client.post(
        "/agent_chat",
        json={"message": "hello", "max_iterations": 3},
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "agent reply"}
    assert calls == [("hello", 3)]


def test_langchain_agent_chat_calls_langchain_agent_loop(monkeypatch):
    calls = []

    def fake_run_langchain_agent_loop(user_input: str, max_iterations: int = 5) -> str:
        calls.append((user_input, max_iterations))
        return "langchain agent reply"

    monkeypatch.setattr(main, "run_langchain_agent_loop", fake_run_langchain_agent_loop)
    client = TestClient(main.app)

    response = client.post(
        "/langchain_agent_chat",
        json={"message": "hello", "max_iterations": 3},
    )

    assert response.status_code == 200
    assert response.json() == {"reply": "langchain agent reply"}
    assert calls == [("hello", 3)]


def test_langgraph_agent_chat_calls_langgraph_agent_loop(monkeypatch):
    calls = []

    def fake_run_langgraph_agent_loop(
        user_input: str,
        max_iterations: int = 5,
        thread_id: str = "default",
    ):
        calls.append((user_input, max_iterations, thread_id))
        return main.LangGraphAgentChatResponse(
            reply="langgraph agent reply",
            messages=[{"type": "human", "content": "hello"}],
        )

    monkeypatch.setattr(main, "run_langgraph_agent_loop", fake_run_langgraph_agent_loop)
    client = TestClient(main.app)

    response = client.post(
        "/langgraph_agent_chat",
        json={"message": "hello", "max_iterations": 3, "thread_id": "thread-1"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "langgraph agent reply",
        "messages": [{"type": "human", "content": "hello"}],
    }
    assert calls == [("hello", 3, "thread-1")]


def test_research_agent_chat_calls_research_agent(monkeypatch):
    calls = []

    def fake_run_research_agent(
        question: str,
        max_rounds: int = 2,
        top_k: int = 5,
        thread_id: str = "default",
    ):
        calls.append((question, max_rounds, top_k, thread_id))
        return main.ResearchAgentChatResponse(
            reply="research reply",
            plan=["q1", "q2"],
            retrievals=[{"query": "q1", "hits": []}],
            missing_points=[],
            citations=[{"source": "docs/a.md", "chunk_id": "c1"}],
        )

    monkeypatch.setattr(main, "run_research_agent", fake_run_research_agent)
    client = TestClient(main.app)

    response = client.post(
        "/research_agent_chat",
        json={
            "message": "complex question",
            "max_rounds": 2,
            "top_k": 4,
            "thread_id": "thread-1",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "research reply",
        "plan": ["q1", "q2"],
        "retrievals": [{"query": "q1", "hits": []}],
        "missing_points": [],
        "citations": [{"source": "docs/a.md", "chunk_id": "c1"}],
    }
    assert calls == [("complex question", 2, 4, "thread-1")]


def test_chat_auto_routes_tool_questions_to_langgraph_agent(monkeypatch):
    calls = []

    def fake_run_langgraph_agent_loop(
        user_input: str,
        max_iterations: int = 5,
        thread_id: str = "default",
    ):
        calls.append((user_input, max_iterations, thread_id))
        return main.LangGraphAgentChatResponse(
            reply="tool reply",
            messages=[{"type": "human", "content": user_input}],
        )

    monkeypatch.setattr(main, "run_langgraph_agent_loop", fake_run_langgraph_agent_loop)
    client = TestClient(main.app)

    response = client.post(
        "/chat",
        json={"message": "帮我算一下 128 * 37", "thread_id": "demo-thread"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "tool reply",
        "mode": "agent",
        "trace": [{"type": "human", "content": "帮我算一下 128 * 37"}],
        "plan": [],
        "retrievals": [],
        "missing_points": [],
        "citations": [],
    }
    assert calls == [("帮我算一下 128 * 37", 5, "demo-thread")]


def test_chat_auto_routes_research_questions_to_research_agent(monkeypatch):
    calls = []

    def fake_run_research_agent(
        question: str,
        max_rounds: int = 2,
        top_k: int = 5,
        thread_id: str = "default",
    ):
        calls.append((question, max_rounds, top_k, thread_id))
        return main.ResearchAgentChatResponse(
            reply="research reply",
            plan=["q1", "q2"],
            retrievals=[{"query": "q1", "hits": []}],
            missing_points=["missing"],
            citations=[{"source": "docs/a.md", "chunk_id": "c1"}],
        )

    monkeypatch.setattr(main, "run_research_agent", fake_run_research_agent)
    client = TestClient(main.app)

    response = client.post(
        "/chat",
        json={
            "message": "帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系",
            "thread_id": "research-thread",
            "max_rounds": 3,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "reply": "research reply",
        "mode": "research",
        "trace": [],
        "plan": ["q1", "q2"],
        "retrievals": [{"query": "q1", "hits": []}],
        "missing_points": ["missing"],
        "citations": [{"source": "docs/a.md", "chunk_id": "c1"}],
    }
    assert calls == [
        (
            "帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系",
            3,
            4,
            "research-thread",
        )
    ]
