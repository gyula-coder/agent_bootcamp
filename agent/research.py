from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver

from agent.langchain_loop import _build_model
from agent.research_prompts import EVALUATION_PROMPT, PLANNING_PROMPT, SYNTHESIS_PROMPT
from agent.research_state import ResearchAgentResult, ResearchState
from memory.store import MemoryRecord, SQLiteMemoryStore
from tools.rag import retrieve_knowledge


class ResearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)


class EvidenceEvaluation(BaseModel):
    is_sufficient: bool = False
    missing_points: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)


class SynthesizedAnswer(BaseModel):
    answer: str


_memory_store = SQLiteMemoryStore()


def _invoke_structured(chain: Any, payload: dict[str, Any], system_prompt: str) -> Any:
    try:
        return chain.invoke(payload)
    except Exception:
        return chain.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=str(payload)),
            ]
        )


def _clean_queries(queries: list[str], fallback: str, limit: int = 3) -> list[str]:
    cleaned = []
    for query in queries:
        normalized = str(query).strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
        if len(cleaned) >= limit:
            break
    return cleaned or [fallback]


def _should_use_short_term_history(question: str) -> bool:
    follow_up_markers = [
        "继续",
        "刚才",
        "上面",
        "前面",
        "基于",
        "补充",
        "展开",
        "再总结",
        "改成",
        "换成",
    ]
    return any(marker in question for marker in follow_up_markers)


def _memory_record_to_dict(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "type": record.type,
        "content": record.content,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def memory_add(
    content: str,
    type: str,
    namespace: str = "default",
) -> dict[str, Any]:
    return _memory_record_to_dict(_memory_store.add(content, type, namespace))


def memory_search(
    query: str,
    namespace: str = "default",
    limit: int = 5,
) -> list[dict[str, Any]]:
    return [
        _memory_record_to_dict(record)
        for record in _memory_store.search(query, namespace, limit)
    ]


def memory_update(
    id: str,
    content: str,
    namespace: str = "default",
) -> dict[str, Any] | None:
    record = _memory_store.update(id, content, namespace)
    return _memory_record_to_dict(record) if record is not None else None


def _load_long_term_memories(state: ResearchState) -> dict[str, Any]:
    namespace = state.get("memory_namespace", "default")
    memories = memory_search(
        state.get("current_question", ""),
        namespace=namespace,
    )
    if not memories:
        memories = [
            _memory_record_to_dict(record)
            for record in _memory_store.list_recent(namespace=namespace)
        ]
    return {
        "long_term_memories": memories,
    }


def _reset_current_turn(state: ResearchState) -> dict[str, Any]:
    return {
        "current_plan": [],
        "current_retrievals": [],
        "current_citations": [],
        "current_missing_points": [],
        "current_answer": "",
        "long_term_memories": [],
        "pending_queries": [],
        "round": 1,
        "is_sufficient": False,
        "follow_up_queries": [],
    }


def _extract_explicit_memory(question: str) -> tuple[str, str] | None:
    normalized = question.strip()
    prefixes = ["记住：", "记住:", "请记住：", "请记住:"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            content = normalized[len(prefix) :].strip()
            if content:
                return content, _classify_memory_type(content)
    return None


def _classify_memory_type(content: str) -> str:
    if any(marker in content for marker in ["项目", "配置", "检索策略", "top_k"]):
        return "project"
    if any(marker in content for marker in ["喜欢", "偏好", "习惯", "默认"]):
        return "preference"
    return "note"


def _plan_queries(model: Any, state: ResearchState) -> dict[str, Any]:
    planner = model.with_structured_output(ResearchPlan)
    question = state["current_question"]
    try:
        plan = _invoke_structured(
            planner,
            {
                "question": question,
                "long_term_memories": state.get("long_term_memories", []),
            },
            PLANNING_PROMPT,
        )
        queries = _clean_queries(plan.queries, question)
    except Exception:
        queries = [question]

    return {
        "current_plan": queries,
        "pending_queries": queries,
        "round": 1,
    }


def _citation_from_hit(hit: dict[str, Any]) -> dict[str, Any] | None:
    source = str(hit.get("source", "")).strip()
    chunk_id = str(hit.get("chunk_id", "")).strip()
    if not source and not chunk_id:
        return None
    return {
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": hit.get("chunk_index"),
        "score": hit.get("score"),
    }


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for citation in citations:
        key = (citation.get("source"), citation.get("chunk_id"), citation.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _retrieve_pending_queries(state: ResearchState) -> dict[str, Any]:
    retrievals = list(state.get("current_retrievals", []))
    citations = list(state.get("current_citations", []))
    top_k = int(state.get("top_k", 5))

    for query in state.get("pending_queries", []):
        try:
            hits = retrieve_knowledge(query=query, top_k=top_k, strategy="adaptive")
            if not isinstance(hits, list):
                hits = []
            retrievals.append({"query": query, "hits": hits})
            for hit in hits:
                if isinstance(hit, dict):
                    citation = _citation_from_hit(hit)
                    if citation is not None:
                        citations.append(citation)
        except Exception as exc:
            retrievals.append({"query": query, "hits": [], "error": str(exc)})

    return {
        "current_retrievals": retrievals,
        "current_citations": _dedupe_citations(citations),
        "pending_queries": [],
    }


def _evaluate_evidence(model: Any, state: ResearchState) -> dict[str, Any]:
    evaluator = model.with_structured_output(EvidenceEvaluation)
    try:
        evaluation = _invoke_structured(
            evaluator,
            {
                "question": state["current_question"],
                "planned_queries": state.get("current_plan", []),
                "retrievals": state.get("current_retrievals", []),
                "missing_points": state.get("current_missing_points", []),
            },
            EVALUATION_PROMPT,
        )
        follow_up_queries = _clean_queries(
            evaluation.follow_up_queries,
            state["current_question"],
            limit=3,
        )
        if evaluation.is_sufficient:
            follow_up_queries = []
        return {
            "is_sufficient": evaluation.is_sufficient,
            "current_missing_points": evaluation.missing_points,
            "follow_up_queries": follow_up_queries,
        }
    except Exception:
        return {
            "is_sufficient": False,
            "current_missing_points": ["无法可靠评估当前证据是否完整。"],
            "follow_up_queries": [],
        }


def _should_continue_research(state: ResearchState) -> str:
    if state.get("is_sufficient"):
        return "synthesize"
    if int(state.get("round", 1)) >= int(state.get("max_rounds", 2)):
        return "synthesize"
    if not state.get("follow_up_queries"):
        return "synthesize"
    return "retrieve"


def _prepare_next_round(state: ResearchState) -> dict[str, Any]:
    return {
        "pending_queries": state.get("follow_up_queries", []),
        "round": int(state.get("round", 1)) + 1,
    }


def _synthesize_answer(model: Any, state: ResearchState) -> dict[str, Any]:
    synthesizer = model.with_structured_output(SynthesizedAnswer)
    use_history = state.get("use_short_term_history", False)
    try:
        answer = _invoke_structured(
            synthesizer,
            {
                "question": state["current_question"],
                "messages": state.get("messages", []) if use_history else [],
                "research_history": state.get("research_history", []) if use_history else [],
                "long_term_memories": state.get("long_term_memories", []),
                "planned_queries": state.get("current_plan", []),
                "retrievals": state.get("current_retrievals", []),
                "missing_points": state.get("current_missing_points", []),
                "citations": state.get("current_citations", []),
            },
            SYNTHESIS_PROMPT,
        )
        return {"current_answer": answer.answer}
    except Exception:
        return {
            "current_answer": "当前无法生成完整结构化答案；请查看检索轨迹和证据缺口。"
        }


def _record_research_turn(state: ResearchState) -> dict[str, Any]:
    update: dict[str, Any] = {
        "research_history": list(state.get("research_history", []))
        + [
            {
                "question": state.get("current_question", ""),
                "citations": state.get("current_citations", []),
                "missing_points": state.get("current_missing_points", []),
                "answer": state.get("current_answer", ""),
            }
        ]
    }
    explicit_memory = _extract_explicit_memory(state.get("current_question", ""))
    if explicit_memory is not None:
        content, memory_type = explicit_memory
        memory_add(
            content=content,
            type=memory_type,
            namespace=state.get("memory_namespace", "default"),
        )
    return update

_checkpointer = MemorySaver()
def build_research_graph(model: Any):
    graph = StateGraph(ResearchState)
    graph.add_node("reset_current_turn", _reset_current_turn)
    graph.add_node("load_long_term_memories", _load_long_term_memories)
    graph.add_node("plan_queries", lambda state: _plan_queries(model, state))
    graph.add_node("retrieve", _retrieve_pending_queries)
    graph.add_node("evaluate_evidence", lambda state: _evaluate_evidence(model, state))
    graph.add_node("prepare_next_round", _prepare_next_round)
    graph.add_node("synthesize", lambda state: _synthesize_answer(model, state))
    graph.add_node("record_turn", _record_research_turn)

    graph.add_edge(START, "reset_current_turn")
    graph.add_edge("reset_current_turn", "load_long_term_memories")
    graph.add_edge("load_long_term_memories", "plan_queries")
    graph.add_edge("plan_queries", "retrieve")
    graph.add_edge("retrieve", "evaluate_evidence")
    graph.add_conditional_edges(
        "evaluate_evidence",
        _should_continue_research,
        {
            "retrieve": "prepare_next_round",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("prepare_next_round", "retrieve")
    graph.add_edge("synthesize", "record_turn")
    graph.add_edge("record_turn", END)
    return graph.compile(checkpointer=_checkpointer)


def run_research_agent(
    question: str,
    max_rounds: int = 2,
    top_k: int = 5,
    thread_id: str = "default",
) -> ResearchAgentResult:
    model = _build_model()
    state: ResearchState = {
        "messages": [HumanMessage(content=question)],
        "current_question": question,
        "use_short_term_history": _should_use_short_term_history(question),
        "max_rounds": max(1, min(max_rounds, 5)),
        "top_k": max(1, min(top_k, 10)),
        "current_retrievals": [],
        "current_citations": [],
        "current_missing_points": [],
        "memory_namespace": "default",
    }

    config = {
        "configurable": {"thread_id": thread_id},
    }
    state = build_research_graph(model).invoke(state, config=config)

    return ResearchAgentResult(
        reply=state.get("current_answer", ""),
        plan=state.get("current_plan", []),
        retrievals=state.get("current_retrievals", []),
        missing_points=state.get("current_missing_points", []),
        citations=state.get("current_citations", []),
    )
