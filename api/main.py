from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.graph import run_langgraph_agent_loop
from agent.langchain_loop import run_langchain_agent_loop
from agent.loop import run_agent_loop
from agent.research import run_research_agent


app = FastAPI(title="Agent Bootcamp API")


class AgentChatRequest(BaseModel):
    message: str
    max_iterations: int = Field(default=5, ge=1)
    thread_id: str = "default"


class AgentChatResponse(BaseModel):
    reply: str


class LangGraphAgentChatResponse(BaseModel):
    reply: str
    messages: list[dict]


class ResearchAgentChatRequest(BaseModel):
    message: str
    max_rounds: int = Field(default=2, ge=1, le=5)
    top_k: int = Field(default=5, ge=1, le=10)
    thread_id: str = "default"


class ResearchAgentChatResponse(BaseModel):
    reply: str
    plan: list[str]
    retrievals: list[dict[str, Any]]
    missing_points: list[str]
    citations: list[dict[str, Any]]


class ChatRequest(BaseModel):
    message: str
    mode: str = Field(default="auto", pattern="^(auto|agent|research)$")
    max_iterations: int = Field(default=5, ge=1)
    max_rounds: int = Field(default=2, ge=1, le=5)
    top_k: int = Field(default=5, ge=1, le=10)
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    mode: str
    trace: list[dict[str, Any]] = []
    plan: list[str] = []
    retrievals: list[dict[str, Any]] = []
    missing_points: list[str] = []
    citations: list[dict[str, Any]] = []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent_chat")
async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    reply = run_agent_loop(
        request.message,
        max_iterations=request.max_iterations,
    )
    return AgentChatResponse(reply=reply)


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    mode = _resolve_chat_mode(request.message, request.mode)
    if mode == "research":
        result = run_research_agent(
            request.message,
            max_rounds=request.max_rounds,
            top_k=request.top_k,
            thread_id=request.thread_id,
        )
        return ChatResponse(
            reply=result.reply,
            mode=mode,
            plan=result.plan,
            retrievals=result.retrievals,
            missing_points=result.missing_points,
            citations=result.citations,
        )

    result = run_langgraph_agent_loop(
        request.message,
        max_iterations=request.max_iterations,
        thread_id=request.thread_id,
    )
    return ChatResponse(
        reply=result.reply,
        mode=mode,
        trace=result.messages,
    )


def _resolve_chat_mode(message: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    research_markers = [
        "梳理",
        "关系",
        "分别解决",
        "从 RAG 工程角度",
        "证据",
        "引用",
        "综合",
        "对比",
        "多次检索",
    ]
    if any(marker in message for marker in research_markers):
        return "research"
    return "agent"

@app.post("/langchain_agent_chat")
async def langchain_agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    reply = run_langchain_agent_loop(
        request.message,
        max_iterations=request.max_iterations,
    )
    return AgentChatResponse(reply=reply)


@app.post("/langgraph_agent_chat")
async def langgraph_agent_chat(
    request: AgentChatRequest,
) -> LangGraphAgentChatResponse:
    result = run_langgraph_agent_loop(
        request.message,
        max_iterations=request.max_iterations,
        thread_id=request.thread_id,
    )
    return LangGraphAgentChatResponse(
        reply=result.reply,
        messages=result.messages,
    )


@app.post("/research_agent_chat")
async def research_agent_chat(
    request: ResearchAgentChatRequest,
) -> ResearchAgentChatResponse:
    result = run_research_agent(
        request.message,
        max_rounds=request.max_rounds,
        top_k=request.top_k,
        thread_id=request.thread_id,
    )
    return ResearchAgentChatResponse(
        reply=result.reply,
        plan=result.plan,
        retrievals=result.retrievals,
        missing_points=result.missing_points,
        citations=result.citations,
    )
