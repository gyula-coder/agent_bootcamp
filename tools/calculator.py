import ast
import operator
from typing import Any


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def calculator(expression: str) -> dict[str, Any]:
    try:
        result = _eval_node(ast.parse(expression, mode="eval").body)
    except Exception as exc:
        return {"ok": False, "error": f"Invalid expression: {exc}"}

    return {"ok": True, "expression": expression, "result": result}


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        operand = _eval_node(node.operand)
        return _OPERATORS[type(node.op)](operand)

    raise ValueError(f"unsupported syntax: {type(node).__name__}")
