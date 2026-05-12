from collections.abc import Callable
from typing import Any

from tools.calculator import calculator
from tools.rag import list_collections, retrieve_knowledge
from tools.time import get_current_time


ToolFunction = Callable[..., Any]


tools: dict[str, ToolFunction] = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "retrieve_knowledge": retrieve_knowledge,
    "list_collections": list_collections,
}


tool_schemas: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate, such as '2 + 3 * 4'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Retrieve relevant mock knowledge chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question or search query for the knowledge base.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of chunks to return.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": "List available mock knowledge collections.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
]
