# Day 4 Log：RAG 工具接入

## 完成内容

1. 将 `retrieve_knowledge(query, top_k, strategy)` 接到已有 RAG 服务的 `POST /retrieve/rerank` 接口。
2. 默认使用 `adaptive` 检索策略，并向 RAG 服务传入 `query`、`top_k`、`chunking_strategy` 和 `threshold`。
3. 将 RAG 返回结果标准化为 Agent 易读的 chunk 列表，字段包含 `content`、`source`、`chunk_id`、`chunk_index`、`score`。
4. 在 LangGraph agent 节点调用模型前临时加入 system message，引导 Agent：
   - 知识库相关问题主动调用 `retrieve_knowledge`。
   - 普通闲聊、问候、简单通用问题不强行调用 RAG。
   - 使用 RAG 回答时体现检索来源。
   - 检索失败或无结果时用自然语言说明，不编造知识库内容。
5. `/langgraph_agent_chat` 现在直接返回 LangGraph 最后一条 AI 消息，避免结构化输出二次总结时丢失引用来源。

## 测试结果

已补充和更新测试：

1. `tests/test_langgraph_loop.py`
   - 验证 LangGraph 会在模型输入前临时加入 system message。
   - 验证工具调用链路仍能返回完整 trace。
   - 验证 RAG 工具结果中的 `source` 能被最终回答保留。
2. `tests/test_rag_tool.py`
   - 验证 RAG 工具向已有 RAG 服务发送 `top_k` 和默认 `adaptive` 策略。
   - 验证检索结果会被标准化为包含 `source`、`chunk_id`、`score` 的结构。

执行命令：

```bash
conda run -n agent-bootcamp python -m pytest tests/test_langgraph_loop.py tests/test_rag_tool.py
```

结果：

```text
3 passed
```

## 当前边界

1. RAG 服务地址暂时写死为 `http://127.0.0.1:8000/retrieve/rerank`，后续可以抽到配置中。
2. 是否调用 RAG 主要由模型根据 tool description 和 system message 判断，适合 Day4 阶段；后续如果需要更稳定的路由，可以增加显式分类节点。
3. 检索失败时，工具异常会进入 tool message，最终错误说明由 Agent 根据 system message 生成。
