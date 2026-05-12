from agent.research_state import ResearchAgentResult
from agent import research
from memory.store import SQLiteMemoryStore


def test_research_agent_result_defaults():
    result = ResearchAgentResult(reply="answer")

    assert result.reply == "answer"
    assert result.plan == []
    assert result.retrievals == []
    assert result.missing_points == []
    assert result.citations == []


def test_plan_queries_uses_model_and_sets_pending_queries(monkeypatch):
    class FakeStructuredModel:
        def invoke(self, payload):
            assert "Transformer" in payload["question"]
            return research.ResearchPlan(
                queries=[
                    "Transformer attention 解决什么问题",
                    "Transformer position encoding 解决什么问题",
                    "RoPE 解决什么问题",
                ]
            )

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is research.ResearchPlan
            return FakeStructuredModel()

    state = {
        "current_question": "帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系",
        "max_rounds": 2,
        "top_k": 5,
    }

    result = research._plan_queries(FakeModel(), state)

    assert result["current_plan"] == [
        "Transformer attention 解决什么问题",
        "Transformer position encoding 解决什么问题",
        "RoPE 解决什么问题",
    ]
    assert result["pending_queries"] == result["current_plan"]
    assert result["round"] == 1


def test_retrieve_pending_queries_calls_rag_for_each_query(monkeypatch):
    calls = []

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        calls.append((query, top_k, strategy))
        return [
            {
                "chunk_id": f"{query}-1",
                "content": f"{query} content",
                "source": "docs/transformer.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    state = {
        "pending_queries": ["attention", "RoPE"],
        "current_retrievals": [],
        "current_citations": [],
        "top_k": 3,
    }

    result = research._retrieve_pending_queries(state)

    assert calls == [
        ("attention", 3, "adaptive"),
        ("RoPE", 3, "adaptive"),
    ]
    assert [item["query"] for item in result["current_retrievals"]] == ["attention", "RoPE"]
    assert result["pending_queries"] == []
    assert result["current_citations"] == [
        {
            "source": "docs/transformer.md",
            "chunk_id": "attention-1",
            "chunk_index": 1,
            "score": 0.9,
        },
        {
            "source": "docs/transformer.md",
            "chunk_id": "RoPE-1",
            "chunk_index": 1,
            "score": 0.9,
        },
    ]


def test_evaluate_evidence_sets_follow_up_queries_when_insufficient():
    class FakeStructuredModel:
        def invoke(self, payload):
            assert payload["question"] == "Transformer components"
            assert payload["retrievals"]
            return research.EvidenceEvaluation(
                is_sufficient=False,
                missing_points=["缺少 RoPE 和 position encoding 的关系"],
                follow_up_queries=["RoPE 和 position encoding 的关系"],
            )

    class FakeModel:
        def with_structured_output(self, schema):
            assert schema is research.EvidenceEvaluation
            return FakeStructuredModel()

    state = {
        "current_question": "Transformer components",
        "current_retrievals": [{"query": "attention", "hits": [{"content": "attention"}]}],
    }

    result = research._evaluate_evidence(FakeModel(), state)

    assert result["is_sufficient"] is False
    assert result["current_missing_points"] == ["缺少 RoPE 和 position encoding 的关系"]
    assert result["follow_up_queries"] == ["RoPE 和 position encoding 的关系"]


def test_prepare_next_round_uses_follow_up_queries_until_max_rounds():
    state = {
        "is_sufficient": False,
        "round": 1,
        "max_rounds": 2,
        "follow_up_queries": ["RoPE 和 position encoding 的关系"],
    }

    assert research._should_continue_research(state) == "retrieve"
    assert research._prepare_next_round(state) == {
        "pending_queries": ["RoPE 和 position encoding 的关系"],
        "round": 2,
    }


def test_run_research_agent_calls_retrieve_multiple_times_and_returns_citations(monkeypatch):
    retrieve_calls = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(
                    queries=[
                        "attention 解决什么问题",
                        "position encoding 解决什么问题",
                    ]
                )
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                return research.SynthesizedAnswer(
                    answer="## 综合结论\nattention 与 position encoding 分工不同。\n\n## 引用来源\n- docs/transformer.md"
                )
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        retrieve_calls.append(query)
        return [
            {
                "chunk_id": query,
                "content": f"{query} content",
                "source": "docs/transformer.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    result = research.run_research_agent(
        "帮我梳理 Transformer 中 attention、position encoding 的关系",
        max_rounds=2,
        top_k=5,
    )

    assert retrieve_calls == [
        "attention 解决什么问题",
        "position encoding 解决什么问题",
    ]
    assert "综合结论" in result.reply
    assert result.plan == [
        "attention 解决什么问题",
        "position encoding 解决什么问题",
    ]
    assert result.citations == [
        {
            "source": "docs/transformer.md",
            "chunk_id": "attention 解决什么问题",
            "chunk_index": 1,
            "score": 0.9,
        },
        {
            "source": "docs/transformer.md",
            "chunk_id": "position encoding 解决什么问题",
            "chunk_index": 1,
            "score": 0.9,
        },
    ]


def test_run_research_agent_retrieves_follow_up_queries_when_evidence_is_insufficient(monkeypatch):
    retrieve_calls = []
    evaluations = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(queries=["attention 解决什么问题"])
            if self.schema is research.EvidenceEvaluation:
                evaluations.append(payload)
                if len(evaluations) == 1:
                    return research.EvidenceEvaluation(
                        is_sufficient=False,
                        missing_points=["缺少 RoPE 关系"],
                        follow_up_queries=["RoPE 和 position encoding 的关系"],
                    )
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                return research.SynthesizedAnswer(answer="含补充检索的答案")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        retrieve_calls.append(query)
        return [
            {
                "chunk_id": query,
                "content": f"{query} content",
                "source": "docs/transformer.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    result = research.run_research_agent(
        "帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系",
        max_rounds=2,
        top_k=5,
    )

    assert retrieve_calls == [
        "attention 解决什么问题",
        "RoPE 和 position encoding 的关系",
    ]
    assert result.reply == "含补充检索的答案"
    assert result.missing_points == []


def test_run_research_agent_keeps_history_but_does_not_use_it_for_independent_questions(monkeypatch):
    synthesis_payloads = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(queries=[f"{payload['question']} query"])
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                synthesis_payloads.append(payload)
                return research.SynthesizedAnswer(answer=f"answer {len(synthesis_payloads)}")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        return [
            {
                "chunk_id": query,
                "content": f"{query} content",
                "source": "docs/short-term.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    research.run_research_agent("first question", thread_id="research-independent-test")
    research.run_research_agent("second question", thread_id="research-independent-test")

    second_payload = synthesis_payloads[1]
    assert second_payload["messages"] == []
    assert second_payload["research_history"] == []


def test_run_research_agent_uses_short_term_history_for_follow_up_questions(monkeypatch):
    synthesis_payloads = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(queries=[f"{payload['question']} query"])
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                synthesis_payloads.append(payload)
                return research.SynthesizedAnswer(answer=f"answer {len(synthesis_payloads)}")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        return [
            {
                "chunk_id": query,
                "content": f"{query} content",
                "source": "docs/short-term.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    research.run_research_agent("first question", thread_id="research-follow-up-test")
    research.run_research_agent("继续刚才的问题，补充一个总结", thread_id="research-follow-up-test")

    second_payload = synthesis_payloads[1]
    assert [message.content for message in second_payload["messages"]] == [
        "first question",
        "继续刚才的问题，补充一个总结",
    ]
    assert second_payload["research_history"][0]["question"] == "first question"
    assert second_payload["research_history"][0]["answer"] == "answer 1"
    assert "retrievals" not in second_payload["research_history"][0]


def test_run_research_agent_resets_current_turn_state_when_checkpoint_resumes(monkeypatch):
    load_states = []
    planner_payloads = []
    evaluation_payloads = []
    synthesis_payloads = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                planner_payloads.append(payload)
                return research.ResearchPlan(queries=[f"{payload['question']} query"])
            if self.schema is research.EvidenceEvaluation:
                evaluation_payloads.append(payload)
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                synthesis_payloads.append(payload)
                return research.SynthesizedAnswer(answer=f"answer {len(synthesis_payloads)}")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    def fake_retrieve_knowledge(query, top_k=5, strategy="adaptive"):
        return [
            {
                "chunk_id": query,
                "content": f"{query} content",
                "source": "docs/reset.md",
                "chunk_index": 1,
                "score": 0.9,
            }
        ]

    original_load_long_term_memories = research._load_long_term_memories

    def capture_load_long_term_memories(state):
        load_states.append(dict(state))
        return original_load_long_term_memories(state)

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(
        research,
        "_load_long_term_memories",
        capture_load_long_term_memories,
    )
    monkeypatch.setattr(research, "retrieve_knowledge", fake_retrieve_knowledge)

    thread_id = "research-reset-current-turn-test"
    research.run_research_agent("first question", thread_id=thread_id)
    research.run_research_agent("second question", thread_id=thread_id)

    assert load_states[1].get("current_plan", []) == []
    assert load_states[1].get("current_answer", "") == ""
    assert load_states[1].get("current_retrievals", []) == []
    assert load_states[1].get("current_citations", []) == []
    assert load_states[1].get("current_missing_points", []) == []
    assert planner_payloads[1]["question"] == "second question"
    assert evaluation_payloads[1]["planned_queries"] == ["second question query"]
    assert evaluation_payloads[1]["missing_points"] == []
    assert synthesis_payloads[1]["planned_queries"] == ["second question query"]
    assert synthesis_payloads[1]["missing_points"] == []


def test_run_research_agent_stores_explicit_stable_memory(monkeypatch, tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(queries=["agent-bootcamp 默认检索策略"])
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                return research.SynthesizedAnswer(answer="已记住。")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "_memory_store", store)
    monkeypatch.setattr(research, "retrieve_knowledge", lambda **kwargs: [])

    research.run_research_agent(
        "记住：我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。",
        thread_id="memory-store-test",
    )

    memories = store.search("adaptive 检索", namespace="default")
    assert len(memories) == 1
    assert memories[0].type == "project"
    assert memories[0].content == "我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。"


def test_run_research_agent_injects_long_term_memory(monkeypatch, tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    store.add(
        "我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。",
        memory_type="project",
        namespace="default",
    )
    planner_payloads = []
    synthesis_payloads = []

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                planner_payloads.append(payload)
                return research.ResearchPlan(queries=["RAG 分块策略"])
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=True,
                    missing_points=[],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                synthesis_payloads.append(payload)
                return research.SynthesizedAnswer(answer="使用项目默认配置回答。")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "_memory_store", store)
    monkeypatch.setattr(research, "retrieve_knowledge", lambda **kwargs: [])

    research.run_research_agent(
        "从 RAG 工程角度看，分块策略解决什么问题？",
        thread_id="memory-inject-test",
    )

    assert planner_payloads[0]["long_term_memories"][0]["content"] == (
        "我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。"
    )
    assert synthesis_payloads[0]["long_term_memories"][0]["type"] == "project"


def test_run_research_agent_does_not_store_transient_research_process(monkeypatch, tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")

    class FakeStructuredModel:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, payload):
            if self.schema is research.ResearchPlan:
                return research.ResearchPlan(queries=["attention 解决什么问题"])
            if self.schema is research.EvidenceEvaluation:
                return research.EvidenceEvaluation(
                    is_sufficient=False,
                    missing_points=["缺少 RoPE 关系"],
                    follow_up_queries=[],
                )
            if self.schema is research.SynthesizedAnswer:
                return research.SynthesizedAnswer(answer="临时研究答案")
            raise AssertionError("unexpected schema")

    class FakeModel:
        def with_structured_output(self, schema):
            return FakeStructuredModel(schema)

    monkeypatch.setattr(research, "_build_model", lambda: FakeModel())
    monkeypatch.setattr(research, "_memory_store", store)
    monkeypatch.setattr(research, "retrieve_knowledge", lambda **kwargs: [])

    research.run_research_agent(
        "帮我梳理 Transformer 中 attention 和 RoPE 的关系",
        thread_id="memory-transient-test",
    )

    assert store.search("RoPE", namespace="default") == []
