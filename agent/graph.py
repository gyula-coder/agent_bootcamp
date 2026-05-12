from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.langchain_loop import (
    _build_model,
    langchain_tools,
)
from agent.state import AgentState


@dataclass
class LangGraphAgentResult:
    reply: str
    messages: list[dict[str, Any]]


_checkpointer = MemorySaver()


def run_langgraph_agent_loop(
    user_input: str,
    max_iterations: int = 5,
    thread_id: str = "default",
) -> LangGraphAgentResult:
    model = _build_model()
    bound_model = model.bind_tools(langchain_tools)
    graph = _build_graph(bound_model)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_iterations * 2 + 2,
    }

    final_state = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    messages = final_state["messages"]

    return LangGraphAgentResult(
        reply=messages[-1].content,
        messages=[_message_to_trace_item(message) for message in messages],
    )


def _build_graph(bound_model: Any):
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node(bound_model))
    graph.add_node("tools", ToolNode(langchain_tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=_checkpointer)

AGENT_SYSTEM_MESSAGE = """你是一个课程学习助手。
当用户问题可能和线性代数、大模型原理相关时，你应该主动调用 retrieve_knowledge 工具。
当用户只是普通闲聊、问候、简单通用问题时，不要强行调用 RAG 工具。
如果使用了 RAG 工具回答问题，回答中需要体现检索来源。
如果检索失败、没有找到相关内容，或工具返回错误，不要编造知识库内容，要用自然语言说明检索失败或没有找到相关资料。
"""


def _agent_node(bound_model: Any):
    def invoke_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=AGENT_SYSTEM_MESSAGE)] + state["messages"]
        return {"messages": [bound_model.invoke(messages)]}

    return invoke_model


def _route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls:
        return "tools"

    return "end"


def _message_to_trace_item(message: BaseMessage) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        item["tool_calls"] = tool_calls

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        item["tool_call_id"] = tool_call_id

    name = getattr(message, "name", None)
    if name:
        item["name"] = name

    return item
