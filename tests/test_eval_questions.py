import json
from pathlib import Path


def test_eval_questions_has_at_least_ten_valid_items():
    path = Path("eval/questions.jsonl")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(lines) >= 10

    items = [json.loads(line) for line in lines]
    assert all(item["question"].strip() for item in items)
    assert {"chat", "tool", "rag", "research", "memory"} <= {
        item["category"] for item in items
    }
