from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    current_question: str
    current_plan: list[str]
    current_retrievals: list[dict[str, Any]]
    current_citations: list[dict[str, Any]]
    current_missing_points: list[str]
    current_answer: str
    long_term_memories: list[dict[str, Any]]
    memory_namespace: str
    use_short_term_history: bool
    research_history: list[dict[str, Any]]
    pending_queries: list[str]
    round: int
    max_rounds: int
    top_k: int
    is_sufficient: bool
    follow_up_queries: list[str]


@dataclass
class ResearchAgentResult:
    reply: str
    plan: list[str] = field(default_factory=list)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    missing_points: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
