# Agent Bootcamp 7天学习计划 PRD

版本：v1.0
日期：2026-04-30
项目路径：/Users/suoer/Develop/agent-bootcamp

## 1. 背景

RAG 阶段已经完成了文档入库、分块策略、Embedding 检索、BM25 + Vector 混合检索、评测集与召回指标等核心能力。下一阶段目标是从“RAG 问答系统”升级为“能调用工具、观察结果、持续决策的 Agent 系统”。

本项目不从大型开源 Agent 项目源码开始，而是先用 7 天完成一个可运行、可解释、可复盘的 Agent v1。完成后再进入 LangGraph examples、pydantic-ai、OpenAI Agents SDK、CrewAI/AutoGen、SWE-agent/OpenDevin/Claude Code 类项目阅读。

## 2. 产品目标

7 天结束后，完成一个本地可演示的 Agent v1：

1. 有 FastAPI `/chat` 接口。
2. 有 LangGraph Agent 状态机。
3. 能调用多个工具。
4. 能接入已有 RAG 检索能力。
5. 能进行多轮 tool calling。
6. 能保存会话状态或 checkpoint。
7. 有基础评测问题集。
8. 有清晰项目结构、README 和阶段总结文档。

## 3. 非目标

本阶段暂不做：

1. Multi-Agent。
2. 大型开源项目完整源码阅读。
3. 复杂权限系统。
4. 沙箱执行环境。
5. MCP 深度集成。
6. 多模型路由。
7. Agent 自动改代码。
8. 企业级 tracing 平台。
9. 复杂前端。

## 4. 技术路线

学习和实现顺序：

1. 原生 Tool Calling：先手写最小 Agent Loop，理解底层机制。
2. LangChain 基础组件：只学习工具、模型、Prompt、结构化输出、Retriever 等必要模块。
3. LangGraph：重点学习，用状态机方式实现 Agent Loop、工具调用、Checkpoint 和任务恢复。
4. RAG 工具化：把已有 RAG 服务作为 Agent 的知识工具。
5. Research Agent：支持多次检索、证据合并和结构化回答。

推荐精力分配：

- LangGraph：60%
- LangChain 基础：20%
- 原生 API Tool Calling：20%

## 5. 建议项目结构

```text
agent-bootcamp/
  api/
    main.py
  agent/
    graph.py
    state.py
    tools.py
    prompts.py
    loop.py
  tools/
    rag.py
    calculator.py
    time.py
  memory/
    checkpoint.py
  eval/
    questions.jsonl
  docs/
    agent-7-day-learning-prd.md
    day1-log.md
    day2-log.md
    day3-log.md
    day4-log.md
    day5-log.md
    day6-log.md
    day7-summary.md
  README.md
```

## 6. 功能范围

### 6.1 Agent Loop

必须支持：

1. 接收用户输入。
2. 调用模型并传入 tools。
3. 判断模型是否返回 tool_calls。
4. 解析 tool name 和 arguments。
5. 执行本地 Python 工具函数。
6. 把工具结果追加回 messages。
7. 再次调用模型。
8. 如果没有 tool_calls，输出最终答案。
9. 通过 max_iterations 防止无限循环。

### 6.2 工具系统

第一阶段工具：

1. `calculator(expression)`：计算表达式。
2. `get_current_time()`：获取当前时间。
3. `retrieve_knowledge(query, top_k)`：检索知识库，Day1 可先 mock，Day4 接真实 RAG 服务。
4. `list_collections()`：列出知识库 collection。

工具要求：

1. 每个工具有清晰 description。
2. 每个工具有参数 schema。
3. 参数错误不能导致程序崩溃。
4. 未知工具名要返回清晰错误。
5. 工具执行异常要被捕获并返回给 Agent。

### 6.3 RAG 工具接入

Day4 起接入已有 RAG 项目检索接口。

要求：

1. 封装 `retrieve_knowledge(query, top_k, strategy)`。
2. 默认支持 `adaptive` 检索策略。
3. 工具返回内容包含 chunk 内容、source、chunk_id、score 或 rank。
4. Agent 回答知识库相关问题时应主动调用 RAG 工具。
5. 普通闲聊问题不应强行调用 RAG 工具。
6. 检索失败时给出可理解错误说明。

### 6.4 LangGraph 状态机

Day3 起实现最小状态图：

```text
START
  -> agent_node
  -> 如果需要 tool，进入 tools_node
  -> tools_node 返回 agent_node
  -> 如果不需要 tool，END
```

要求：

1. State 至少包含 `messages`。
2. `agent_node` 负责调用模型。
3. `tools_node` 负责执行工具。
4. conditional edges 根据 tool_calls 决定继续或结束。
5. 支持基础 checkpoint 或 thread_id 级别会话恢复。

## 7. 7天里程碑

### Day 1：手写 Tool Calling Loop

目标：不用 LangChain，不用 LangGraph，先把工具调用底层机制打透。

要完成：

1. 定义 tools 字典。
2. 定义 tool schema。
3. 调用模型并传入 tools。
4. 判断模型是否返回 tool_calls。
5. 解析 tool name 和 arguments。
6. 执行本地 Python 函数。
7. 把 tool result 追加回 messages。
8. 再次调用模型。
9. 如果没有 tool_calls，则输出最终答案。
10. 加 max_iterations，防止死循环。

建议工具：

- `calculator(expression)`
- `get_current_time()`
- `retrieve_knowledge(query, top_k)`
- `list_collections()`

验收标准：

1. 能完成一次“用户问题 -> 模型选择工具 -> 执行工具 -> 模型基于结果回答”的闭环。
2. 工具参数错误时能返回清晰错误信息。
3. Agent 不会无限循环。
4. `docs/day1-log.md` 记录完成项、问题与修复。

### Day 2：用 LangChain 标准化工具和模型调用

目标：把 Day1 的手写版本改成更标准的 LangChain 组件写法。

要学习：

1. `@tool`
2. Pydantic 参数 schema
3. `ChatModel.bind_tools()`
4. `tool.invoke()`
5. `ChatPromptTemplate`
6. structured output
7. 基础 LCEL：`prompt | model | parser`

要完成：

1. 用 `@tool` 重写 Day1 的工具。
2. 给每个工具增加清晰 description。
3. 用 Pydantic 限制工具参数。
4. 把模型调用统一封装到一个模块中。
5. 保留 Day1 的 Agent Loop，但工具定义改为 LangChain 风格。

验收标准：

1. 工具 schema 清晰可读。
2. 模型能正确选择工具。
3. 参数校验失败时不会导致程序崩溃。
4. 代码结构比 Day1 更清晰。
5. `docs/day2-log.md` 记录改造内容和对比结论。

### Day 3：用 LangGraph 实现最小 Agent 状态机

目标：把 Agent Loop 改造成 LangGraph 状态机。

要学习：

1. `StateGraph`
2. `START` / `END`
3. `add_node`
4. `add_edge`
5. `add_conditional_edges`
6. `ToolNode`
7. `MemorySaver`

要完成：

1. 定义 Agent State，至少包含 `messages`。
2. 实现 `agent_node`：调用模型。
3. 实现 `tools_node`：执行工具。
4. 实现条件边：根据是否存在 tool_calls 决定继续或结束。
5. 加入基础 checkpoint。

验收标准：

1. 同一个问题可以经过多轮工具调用。
2. 状态能在节点之间正确传递。
3. 能看到每一步 messages 的变化。
4. 支持 checkpoint 或 thread_id 级别的会话恢复。
5. `docs/day3-log.md` 记录图结构、执行轨迹和问题。

### Day 4：接入已有 RAG 检索工具

目标：把 RAG 从“独立问答服务”变成 Agent 的知识工具。

要完成：

1. 封装 `retrieve_knowledge(query, top_k, strategy)`。
2. 默认接入已有 RAG 项目的检索接口。
3. 支持 `adaptive` 检索策略。
4. 工具返回内容包含 chunk 内容、source、chunk_id、score 或 rank。
5. Agent 回答时必须基于检索证据，并尽量给出引用。

建议工具：

- `retrieve_knowledge(query, top_k, strategy)`
- `list_collections()`
- `get_chunk(chunk_id)`

验收标准：

1. 用户问知识库相关问题时，Agent 会主动调用 RAG 工具。
2. 用户问普通闲聊问题时，Agent 不应强行调用 RAG 工具。
3. 回答中能体现检索来源。
4. 检索失败时能给出可理解的错误说明。
5. `docs/day4-log.md` 记录 RAG 接入方式和测试结果。

### Day 5：做 Research Agent

目标：让 Agent 不只是一次检索，而是能围绕复杂问题多次检索、合并证据、形成结构化答案。

要完成：

1. 让 Agent 对复杂问题拆成多个子问题。
2. 对每个子问题调用 RAG 检索。
3. 合并不同检索结果。
4. 发现证据不足时继续检索。
5. 输出结构化答案。
6. 附带引用来源。

测试问题示例：

```text
帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系，并指出它们分别解决什么问题。
```

验收标准：

1. 至少能对一个复杂问题发起 2 次以上检索。
2. 最终答案不是简单拼接 chunks，而是有归纳结构。
3. 能明确指出信息不足或证据缺口。
4. 有引用来源。
5. `docs/day5-log.md` 记录复杂问题拆解、检索轨迹和最终答案。

### Day 6：加入 Memory / Checkpoint

目标：让 Agent 从单轮工具调用升级成可恢复、有状态的任务系统。

要完成：

1. 使用 LangGraph checkpointer 保存会话状态。
2. 支持 thread_id。
3. 区分短期记忆和长期记忆。
4. 设计最小 memory 工具。
5. 记录用户偏好、项目背景、常用配置等稳定信息。

建议工具：

- `memory_search(query)`
- `memory_add(content, type)`
- `memory_update(id, content)`

验收标准：

1. 同一个 thread_id 下，Agent 能接续上下文。
2. Agent 不会把所有临时过程都写入长期记忆。
3. 能解释什么信息应该记、什么信息不应该记。
4. checkpoint 恢复后能继续多轮对话。
5. `docs/day6-log.md` 记录 memory/checkpoint 设计和测试结果。

### Day 7：整理 Agent v1 演示版

目标：冻结一个可展示、可复盘、可继续扩展的 Agent v1。

要完成：

1. 整理 FastAPI `/chat` 接口。
2. 整理工具列表和工具说明。
3. 整理 LangGraph 图结构说明。
4. 增加 eval questions。
5. 写 README。
6. 写 Day1 到 Day7 总结文档。
7. 明确下一阶段是否进入开源项目阅读。

验收标准：

1. 本地能启动服务。
2. `/chat` 能完成至少 3 类问题：普通问答、工具调用、RAG Research。
3. README 能说明架构和运行方式。
4. 有至少 10 条测试问题。
5. 有失败案例和后续改进清单。
6. `docs/day7-summary.md` 完成最终复盘。

## 8. 最终验收清单

项目在 Day7 结束时必须满足：

1. 服务可启动。
2. `/chat` 可调用。
3. 至少 3 个工具可用。
4. 至少 1 个问题触发多轮 tool calling。
5. RAG 工具能接入已有知识库检索。
6. Research Agent 能对复杂问题进行至少 2 次检索。
7. 支持 thread_id 或 checkpoint。
8. 有 eval questions，至少 10 条。
9. README 能让别人按步骤跑起来。
10. docs 下有 Day1-Day7 日志或总结。

## 9. 推荐测试问题

### 普通工具调用

```text
帮我算一下 128 * 37
```

```text
现在是什么时间？
```

### RAG 检索

```text
RAG 中 chunk_size 和 overlap 分别控制什么？
```

```text
BM25 和向量检索分别适合什么场景？
```

### Research Agent

```text
帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系，并指出它们分别解决什么问题。
```

```text
从 RAG 工程角度看，分块策略、混合检索和 rerank 分别解决什么问题？
```

### Memory / Checkpoint

```text
记住：我当前项目叫 agent-bootcamp，默认用 adaptive 检索策略。
```

```text
继续刚才的问题，帮我补充一个总结。
```

## 10. 风险与注意事项

1. 不要一开始直接进入 LangGraph，Day1 必须先手写 loop。
2. 不要把旧 LangChain Agent 作为主线，重点是现代 tool_call + 显式状态机。
3. Day4 前不要过早接复杂 RAG Research，先把基础工具调用稳定下来。
4. Memory 只记录稳定信息，不要把所有临时过程写入长期记忆。
5. Research Agent 重点是多次检索和证据合并，不是简单拼接 chunk。
6. 每天都要有可验收产物，不只写代码。

## 11. 开源项目阅读后置计划

完成 7 天计划后，再按以下顺序阅读开源项目：

1. LangGraph 官方 examples：看标准状态机写法。
2. pydantic-ai examples：看类型安全和结构化 Agent 写法。
3. OpenAI Agents SDK examples：看原生生态标准做法。
4. CrewAI / AutoGen：看多 Agent 设计思想。
5. SWE-agent / OpenDevin / Claude Code 类项目：看真实 Coding Agent 工程形态。

阅读重点：

1. Agent loop 如何实现。
2. Tool schema 如何设计。
3. 状态如何保存。
4. 错误如何恢复。
5. 如何限制权限。
6. 如何记录轨迹和评测。
7. 如何压缩上下文。
