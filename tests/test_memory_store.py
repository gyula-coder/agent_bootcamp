from memory.store import SQLiteMemoryStore


def test_sqlite_memory_store_add_search_and_update(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(db_path)

    memory = store.add(
        content="当前项目叫 agent-bootcamp，默认使用 adaptive 检索策略。",
        memory_type="project",
        namespace="test-user",
    )

    matches = store.search("adaptive 检索", namespace="test-user")
    assert matches == [memory]

    updated = store.update(
        memory.id,
        content="当前项目叫 agent-bootcamp，默认使用 hybrid 检索策略。",
        namespace="test-user",
    )

    assert updated is not None
    assert updated.id == memory.id
    assert updated.content == "当前项目叫 agent-bootcamp，默认使用 hybrid 检索策略。"
    assert updated.updated_at >= memory.updated_at
    assert store.search("adaptive 检索", namespace="test-user") == []
    assert store.search("hybrid 检索", namespace="test-user") == [updated]


def test_sqlite_memory_store_keeps_namespaces_separate(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(db_path)

    store.add("默认 top_k 是 5。", memory_type="preference", namespace="user-a")
    store.add("默认 top_k 是 10。", memory_type="preference", namespace="user-b")

    assert [memory.content for memory in store.search("top_k", namespace="user-a")] == [
        "默认 top_k 是 5。"
    ]
    assert [memory.content for memory in store.search("top_k", namespace="user-b")] == [
        "默认 top_k 是 10。"
    ]


def test_sqlite_memory_store_search_allows_partial_term_matches(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(db_path)
    memory = store.add(
        "我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。",
        memory_type="project",
        namespace="test-user",
    )

    matches = store.search(
        "从 RAG 工程角度看，分块策略解决什么问题？",
        namespace="test-user",
    )

    assert matches == [memory]
