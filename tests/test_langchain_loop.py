import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agent import langchain_loop


def test_execute_langchain_tool_returns_validation_errors():
    result = langchain_loop._execute_langchain_tool(
        {
            "id": "call_1",
            "name": "calculator",
            "args": {},
        }
    )

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    assert result["name"] == "calculator"
    assert '"ok": false' in result["content"]
    assert "Invalid tool arguments" in result["content"]


def test_run_langchain_agent_loop_returns_structured_answer(monkeypatch):
    class FakeModel:
        def bind_tools(self, tools):
            return RunnableLambda(lambda _: AIMessage(content="final answer"))

        def with_structured_output(self, schema):
            assert schema is langchain_loop.AgentStructuredAnswer
            return RunnableLambda(
                lambda _: langchain_loop.AgentStructuredAnswer(
                    summary="一句话总结",
                    details=["要点一", "要点二"],
                    confidence=0.8,
                )
            )

    monkeypatch.setattr(langchain_loop, "_build_model", lambda: FakeModel())

    reply = langchain_loop.run_langchain_agent_loop("hello")

    assert json.loads(reply) == {
        "summary": "一句话总结",
        "details": ["要点一", "要点二"],
        "confidence": 0.8,
    }
