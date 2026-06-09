# Agent Bootcamp

一个 7 天 Agent 学习项目：从手写 tool calling loop，逐步升级到 LangChain 工具、LangGraph 状态机、RAG 工具、Research Agent 和最小 memory/checkpoint。

## 项目结构

```text
api/          FastAPI 接口
agent/        Agent loop、LangGraph、Research Agent
tools/        calculator、time、RAG 工具
memory/       checkpoint 与 SQLite 长期记忆
eval/         Day 7 演示问题集
docs/         PRD、每日日志、最终复盘
tests/        单元测试与接口测试
```

## 安装

本项目使用 conda 环境，建议环境名为 `agent-bootcamp`。

```bash
conda create -n agent-bootcamp python=3.11
conda activate agent-bootcamp
python -m pip install -r requirements.txt
```

模型配置在 `agent/config.py`：

```python
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4.1-mini"
API_KEY = "your-api-key"
```

如果使用 OpenAI-compatible provider，替换这三个值即可。

## 启动服务

```bash
conda run -n agent-bootcamp uvicorn api.main:app --reload --port 8001
```

健康检查：

```bash
curl http://127.0.0.1:8001/health
```

Day 7 统一演示入口：

```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我算一下 128 * 37","thread_id":"demo-1"}'
```

Research Agent 示例：

```bash
curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系，并指出它们分别解决什么问题。","thread_id":"demo-2","mode":"research"}'
```

保留的学习阶段接口：

- `POST /agent_chat`：Day 1 手写 OpenAI tool calling loop。
- `POST /langchain_agent_chat`：Day 2 LangChain 工具版 loop。
- `POST /langgraph_agent_chat`：Day 3/4 LangGraph 工具状态机。
- `POST /research_agent_chat`：Day 5/6 Research Agent + checkpoint + memory。

## 工具列表

- `calculator(expression)`：计算基础四则表达式。
- `get_current_time()`：返回当前本地时间。
- `retrieve_knowledge(query, top_k, strategy)`：调用本地 RAG 检索接口，默认 `adaptive`。
- `list_collections()`：列出可用知识库 collection。
- `memory_add(content, type)` / `memory_search(query)` / `memory_update(id, content)`：Research Agent 使用的最小长期记忆函数。

## LangGraph 图结构

基础 Agent 图：

```text
START -> agent -> tools -> agent -> END
```

`agent` 节点调用绑定工具的模型；如果最后一条 AI 消息包含 `tool_calls`，进入 `tools` 节点执行工具，否则结束。图使用 `MemorySaver`，通过 `thread_id` 支持会话级 checkpoint。

Research Agent 图：

```text
START
  -> reset_current_turn
  -> load_long_term_memories
  -> plan_queries
  -> retrieve
  -> evaluate_evidence
  -> prepare_next_round / synthesize
  -> record_turn
  -> END
```

它会先拆解复杂问题，再多次检索、评估证据是否足够，最后合成带引用和证据缺口的结构化回答。

## 验证

```bash
conda run -n agent-bootcamp python -m pytest
```

Day 7 演示问题集在 `eval/questions.jsonl`，覆盖普通问答、工具调用、RAG、Research Agent 和 memory/checkpoint。
