# Day 3 Log: LangGraph Agent State Machine

## 完成项

1. 新增 `agent/state.py`，定义 `AgentState`，使用 `messages` 作为 LangGraph 状态。
2. 新增 `agent/graph.py`，用 `StateGraph` 实现最小 Agent Loop：

```text
START
  -> agent
  -> tools, if the latest AI message has tool_calls
  -> agent
  -> END, if the latest AI message has no tool_calls
```

3. 复用 Day2 的 LangChain tools，并使用 `ToolNode` 执行工具。
4. 使用 `MemorySaver` 和 `thread_id` 支持基础 checkpoint。
5. 新增 FastAPI 接口 `POST /langgraph_agent_chat`。
6. 接口返回最终 `reply` 和 `messages` 执行轨迹，便于观察每一步状态变化。

## 执行轨迹示例

测试用例 `tests/test_langgraph_loop.py` 覆盖了一次工具调用闭环：

```text
human: 计算 2 + 3
ai: tool_calls=[calculator({"expression": "2 + 3"})]
tool: calculator result
ai: 2 + 3 = 5
```

最终回答会被转换为 Day2 已有的 `AgentStructuredAnswer` JSON 字符串。

## 问题与修复

1. `agent/graph.py` 和 `agent/state.py` 原先是 placeholder，本次替换为真实 LangGraph 实现。
2. API 原先只有 Day1/Day2 接口，本次新增 `/langgraph_agent_chat`，并通过 `thread_id` 传入 LangGraph checkpoint 配置。
3. `requirements.txt` 原先缺少 `langgraph`，本次补充依赖声明。

## 验证

```bash
conda run -n agent-bootcamp python -m pytest
```

结果：8 passed，1 个来自 LangGraph 依赖的 deprecation warning。
