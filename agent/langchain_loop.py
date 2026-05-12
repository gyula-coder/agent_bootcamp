import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent import config
from tools.calculator import calculator as calculate_expression
from tools.rag import list_collections as list_rag_collections
from tools.rag import retrieve_knowledge as retrieve_rag_knowledge
from tools.time import get_current_time as get_local_current_time


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression to evaluate, such as '2 + 3 * 4'.")

class RetrieveKnowledgeArgs(BaseModel):
    query: str = Field(description="Question or search query for the knowledge base.")
    strategy: str = Field(
        default='adaptive',
        description="Strategy for retrieving knowledge."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of chunks to return.",
    )


class EmptyArgs(BaseModel):
    pass


class AgentStructuredAnswer(BaseModel):
    summary: str = Field(description="One-sentence summary of the answer.")
    details: list[str] = Field(description="Detailed answer as a list of key points.")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score from 0 to 1.",
    )

@tool(
    "calculator",
    args_schema=CalculatorArgs,
    description="Calculate a basic arithmetic expression.",
)
def langchain_calculator(expression:str) -> dict[str, Any]:
    return calculate_expression(expression)


@tool(
    "get_current_time",
    args_schema=EmptyArgs,
    description="Get the current local time.",
)
def langchain_get_current_time() -> dict[str, str]:
    return get_local_current_time()


@tool(
    "retrieve_knowledge",
    args_schema=RetrieveKnowledgeArgs,
    description="当用户的问题需要查询线性代数或者大模型原理相关知识时，调用此工具检索相关资料。"
        "如果用户只是闲聊、问通用常识、或问题不依赖本项目知识库，则不要调用。"
        "如果不确定是否相关，优先调用此工具。",
)
def langchain_retrieve_knowledge(query: str, top_k: int = 5, strategy: str = 'adaptive') -> dict[str, Any]:
    return retrieve_rag_knowledge(query=query, top_k=top_k, strategy=strategy)


@tool(
    "list_collections",
    args_schema=EmptyArgs,
    description="List available mock knowledge collections.",
)
def langchain_list_collections() -> dict[str, Any]:
    return list_rag_collections()


langchain_tools: list[BaseTool] = [
    langchain_calculator,
    langchain_get_current_time,
    langchain_retrieve_knowledge,
    langchain_list_collections,
]
tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in langchain_tools}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise agent. Use tools when they help answer the user.",
        ),
        MessagesPlaceholder("messages"),
    ]
)

structured_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Convert the final assistant answer into structured output with a "
            "one-sentence summary, detailed key points, and confidence.",
        ),
        MessagesPlaceholder("messages"),
    ]
)


def run_langchain_agent_loop(user_input: str, max_iterations: int = 5) -> str:
    messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    model = _build_model()
    chain = prompt | model.bind_tools(langchain_tools)
    structured_chain = (
        structured_answer_prompt | model.with_structured_output(AgentStructuredAnswer)
    )

    for _ in range(max_iterations):
        assistant_message = chain.invoke({"messages": messages})
        messages.append(assistant_message)

        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        if not tool_calls:
            structured_answer = structured_chain.invoke({"messages": messages})
            return _structured_answer_to_text(structured_answer)

        for tool_call in tool_calls:
            result = _execute_langchain_tool(tool_call)
            messages.append(
                ToolMessage(
                    content=result["content"],
                    tool_call_id=result["tool_call_id"],
                    name=result["name"],
                )
            )

    return _structured_answer_to_text(
        AgentStructuredAnswer(
            summary="Agent stopped before producing a final answer.",
            details=["The maximum iteration limit was reached."],
            confidence=0,
        )
    )


def _build_model() -> ChatOpenAI:
    if not config.API_KEY:
        raise RuntimeError("API_KEY is required")

    return ChatOpenAI(
        api_key=config.API_KEY,
        base_url=config.BASE_URL,
        model=config.MODEL,
    )


def _execute_langchain_tool(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = tool_call.get("id")
    name = tool_call.get("name")
    arguments = tool_call.get("args") or {}
    selected_tool = tools_by_name.get(name)

    if selected_tool is None:
        result = {"ok": False, "error": f"Unknown tool: {name}"}
    else:
        try:
            result = selected_tool.invoke(arguments)
        except Exception as exc:
            result = {"ok": False, "error": f"Invalid tool arguments: {exc}"}

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _structured_answer_to_text(answer: AgentStructuredAnswer | dict[str, Any]) -> str:
    if isinstance(answer, AgentStructuredAnswer):
        return answer.model_dump_json()

    return json.dumps(answer, ensure_ascii=False)
