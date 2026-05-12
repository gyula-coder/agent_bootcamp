# Day 6 Log: Memory / Checkpoint

## 完成项

1. Research Agent 继续使用 LangGraph `MemorySaver` 作为 thread 级 checkpoint。
2. 增加 SQLite 长期记忆存储：`memory/store.py`。
3. 增加最小 memory 工具函数：
   - `memory_search(query, namespace, limit)`
   - `memory_add(content, type, namespace)`
   - `memory_update(id, content, namespace)`
4. Research Graph 新增 `load_long_term_memories` 节点，在查询规划前加载长期记忆。
5. 合成答案时注入 `long_term_memories`，但明确它只代表稳定背景或用户偏好，不是检索证据。
6. 只对显式“记住：...”类输入写入长期记忆，避免把检索轨迹、证据缺口、临时回答写成长期记忆。

## 记忆边界

应该进入长期记忆的信息：

- 用户偏好：回答语言、输出格式、默认参数。
- 项目背景：项目名、当前技术栈、常用配置。
- 稳定配置：默认检索策略、默认 top_k 等。

不应该进入长期记忆的信息：

- 单次研究问题。
- RAG 检索轨迹。
- chunk 内容和引用列表。
- 临时证据缺口。
- 某次回答的中间过程。

## SQLite 设计

长期记忆表结构：

```sql
memories(
  id TEXT PRIMARY KEY,
  namespace TEXT NOT NULL,
  type TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

当前实现是本地轻量版，接口向 LangGraph Store 的 `namespace + record` 思路靠拢。后续如果要生产化，可以把 SQLite store 替换为 Postgres-backed store 或带 embedding 的语义检索。

## 验证结果

已通过：

```bash
conda run -n agent-bootcamp python -m pytest tests/test_memory_store.py tests/test_research_agent.py
```

结果：14 passed。

覆盖场景：

1. SQLite memory add/search/update。
2. namespace 隔离。
3. 显式稳定信息写入长期记忆。
4. 后续 Research Agent 调用注入长期记忆。
5. 临时研究过程不会写入长期记忆。
