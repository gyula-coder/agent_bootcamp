import json
from typing import Any

from openai import OpenAI

from agent import config
from agent.tools import tool_schemas, tools


Message = dict[str, Any]


def run_agent_loop(user_input: str, max_iterations: int = 5) -> str:
    messages: list[Message] = [{"role": "user", "content": user_input}]

    for _ in range(max_iterations):
        assistant_message = call_model(messages)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return assistant_message.get("content") or ""

        for tool_call in tool_calls:
            messages.append(_execute_tool_call(tool_call))

    return "Agent stopped because max_iterations was reached."


def call_model(messages: list[Message]) -> Message:
    if not config.API_KEY:
        raise RuntimeError("API_KEY is required")

    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    response = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        tools=tool_schemas,
    )
    message = response.choices[0].message
    return message.model_dump(exclude_none=True)


def _execute_tool_call(tool_call: dict[str, Any]) -> Message:
    tool_call_id = tool_call.get("id")
    function = tool_call.get("function") or {}
    name = function.get("name")

    if name not in tools:
        result = {"ok": False, "error": f"Unknown tool: {name}"}
    else:
        try:
            arguments = json.loads(function.get("arguments") or "{}")
            result = tools[name](**arguments)
        except json.JSONDecodeError as exc:
            result = {"ok": False, "error": f"Invalid tool arguments JSON: {exc}"}
        except TypeError as exc:
            result = {"ok": False, "error": f"Invalid tool arguments: {exc}"}
        except Exception as exc:
            result = {"ok": False, "error": f"Tool execution failed: {exc}"}

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }
