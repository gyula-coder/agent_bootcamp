# Day 5 Log：Research Agent

## 完成内容

1. 新增独立 Research Agent，不影响 Day 1-4 普通 Agent。
2. 使用 LangGraph 显式组织 research 流程：问题拆解、批量检索、证据评估、必要时补充检索、最终综合。
3. 复用 Day 4 的 `retrieve_knowledge(query, top_k, strategy)`，默认使用 `adaptive`。
4. 最终响应返回 `reply`、`plan`、`retrievals`、`missing_points` 和 `citations`。
5. 新增 `/research_agent_chat` API。

## 复杂问题拆解示例

问题：

```text
帮我梳理 Transformer 中 attention、position encoding、RoPE 的关系，并指出它们分别解决什么问题。
```

可能拆解：

1. Transformer attention 解决什么问题
2. Transformer position encoding 解决什么问题
3. RoPE rotary position embedding 解决什么问题
4. attention、position encoding、RoPE 三者关系

## 检索轨迹

Research Agent 会对每个子问题调用 RAG 检索，并在证据不足时用 follow-up query 继续检索。API 响应中的 `retrievals` 字段保留每次 query、命中 chunks 和错误信息。

## 最终答案要求

最终答案应包含：

1. 问题拆解
2. 综合结论
3. 分点解释
4. 关系梳理
5. 证据不足或不确定点
6. 引用来源

## 测试结果

执行命令：

```bash
conda run -n agent-bootcamp python -m pytest
```

结果：全部通过。

## 当前边界

1. Day 5 只使用已有 RAG 服务，不接 Web Search。
2. `thread_id` 当前只用于接口兼容，长期记忆留到 Day 6。
3. Research loop 默认最多 2 轮，避免无限检索。
