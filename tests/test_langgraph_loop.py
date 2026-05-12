from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from agent import graph


def test_run_langgraph_agent_loop_executes_tool_and_returns_trace(monkeypatch):
    calls = []

    class FakeModel:
        def bind_tools(self, tools):
            def invoke(messages):
                calls.append([message.type for message in messages])
                if len(calls) == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "calculator",
                                "args": {"expression": "2 + 3"},
                            }
                        ],
                    )

                return AIMessage(content="2 + 3 = 5")

            return RunnableLambda(invoke)

    monkeypatch.setattr(graph, "_build_model", lambda: FakeModel())

    result = graph.run_langgraph_agent_loop(
        "计算 2 + 3",
        thread_id="test-thread",
    )

    assert result.reply == "2 + 3 = 5"
    assert [message["type"] for message in result.messages] == [
        "human",
        "ai",
        "tool",
        "ai",
    ]
    assert calls == [["system", "human"], ["system", "human", "ai", "tool"]]


def test_run_langgraph_agent_loop_preserves_rag_source_in_reply(monkeypatch):
    @tool("retrieve_knowledge")
    def fake_retrieve_knowledge(query: str, top_k: int = 5, strategy: str = "adaptive"):
        """Retrieve relevant knowledge chunks."""
        return [
            {
                "chunk_id": "rag-1",
                "content": "RAG 回答需要体现检索来源。",
                "source": "docs/rag.md",
                "score": 0.91,
            }
        ]

    class FakeModel:
        def bind_tools(self, tools):
            assert [tool.name for tool in tools] == ["retrieve_knowledge"]

            def invoke(messages):
                tool_messages = [message for message in messages if message.type == "tool"]
                if not tool_messages:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "retrieve_knowledge",
                                "args": {"query": "RAG 引用来源", "top_k": 1},
                            }
                        ],
                    )

                assert "docs/rag.md" in tool_messages[-1].content
                return AIMessage(content="根据 docs/rag.md：RAG 回答需要体现检索来源。")

            return RunnableLambda(invoke)

    monkeypatch.setattr(graph, "langchain_tools", [fake_retrieve_knowledge])
    monkeypatch.setattr(graph, "_build_model", lambda: FakeModel())

    result = graph.run_langgraph_agent_loop(
        "RAG 回答要不要带来源？",
        thread_id="test-rag-source-thread",
    )

    assert result.reply == "根据 docs/rag.md：RAG 回答需要体现检索来源。"
    assert [message["type"] for message in result.messages] == [
        "human",
        "ai",
        "tool",
        "ai",
    ]
