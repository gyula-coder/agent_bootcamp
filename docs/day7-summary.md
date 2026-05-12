# Day 7 Summary: Agent v1 Demo

## 完成项

1. 增加统一演示接口 `POST /chat`。
2. 保留 Day1-Day6 学习接口，便于对比演进路径。
3. 整理工具列表：calculator、time、RAG retrieve、collection list、memory functions。
4. 整理 LangGraph 基础 Agent 图和 Research Agent 图结构。
5. 补充 `eval/questions.jsonl`，共 10 条问题，覆盖普通问答、工具调用、RAG、Research、Memory。
6. 重写 README，补充安装、配置、启动、调用、架构和验证方式。
7. 补充 requirements，确保 FastAPI、LangChain、LangGraph、OpenAI client、测试与 HTTP 测试客户端可安装。

## Agent v1 能力

### 普通问答

`/chat` 的 `mode=auto` 默认走 LangGraph Agent。普通寒暄或简单问题不强制调用 RAG。

### 工具调用

LangGraph Agent 支持模型选择工具，并在 `tools` 节点执行：

- `calculator(expression)`
- `get_current_time()`
- `retrieve_knowledge(query, top_k, strategy)`
- `list_collections()`

工具调用错误会被包装为可读错误，避免直接让接口崩溃。

### RAG Research

Research Agent 支持：

1. 复杂问题拆解为多个检索 query。
2. 对每个 query 调用 RAG 检索。
3. 评估证据是否足够。
4. 证据不足时继续检索。
5. 合成结构化答案、引用来源和证据缺口。

### Memory / Checkpoint

基础 LangGraph Agent 和 Research Agent 都使用 `MemorySaver` 支持 `thread_id` 级 checkpoint。Research Agent 额外有 SQLite 长期记忆，只记录显式“记住：...”这类稳定信息，不记录临时检索过程。

## 演示问题

可以从 `eval/questions.jsonl` 选题演示：

1. 普通问答：`你好，请用一句话介绍你能做什么。`
2. 工具调用：`帮我算一下 128 * 37。`
3. RAG：`RAG 中 chunk_size 和 overlap 分别控制什么？`
4. Research：`帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系，并指出它们分别解决什么问题。`
5. Memory：`记住：我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。`

## 失败案例与后续改进

1. 当前 RAG 工具依赖本地 `http://127.0.0.1:8000/retrieve/rerank`，RAG 服务未启动时会返回检索失败。
2. `/chat mode=auto` 只使用轻量关键词路由，复杂场景应改为模型路由或显式前端选择。
3. 长期记忆目前是关键词搜索，不是 embedding 语义检索。
4. 工具权限和执行沙箱还没有做，下一阶段阅读开源项目时需要重点关注。
5. tracing 仅返回 messages/retrievals，不是完整可视化链路。

## 下一阶段

建议进入开源项目阅读，顺序沿用 PRD：

1. LangGraph 官方 examples。
2. pydantic-ai examples。
3. OpenAI Agents SDK examples。
4. CrewAI / AutoGen。
5. SWE-agent / OpenDevin / Claude Code 类项目。

阅读重点：Agent loop、tool schema、状态保存、错误恢复、权限限制、轨迹记录、上下文压缩。

## 验证结果

Day 7 新增验证：

```bash
conda run -n agent-bootcamp python -m pytest tests/test_api.py tests/test_eval_questions.py
```

结果：7 passed。
