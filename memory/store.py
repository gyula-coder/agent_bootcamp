from __future__ import annotations

import sqlite3
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_MEMORY_DB_PATH = Path(__file__).with_name("long_term_memory.sqlite3")


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    namespace: str
    type: str
    content: str
    created_at: str
    updated_at: str


class SQLiteMemoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_MEMORY_DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    def add(
        self,
        content: str,
        memory_type: str,
        namespace: str = "default",
    ) -> MemoryRecord:
        now = _utc_now()
        record = MemoryRecord(
            id=str(uuid4()),
            namespace=namespace,
            type=memory_type,
            content=content.strip(),
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, namespace, type, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.namespace,
                    record.type,
                    record.content,
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def search(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        terms = _search_terms(query)
        if not terms:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, namespace, type, content, created_at, updated_at
                FROM memories
                WHERE namespace = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (namespace,),
            ).fetchall()

        records = [_record_from_row(row) for row in rows]
        required_terms = _required_terms(terms)
        scored_matches = []
        for record in records:
            content = record.content.lower()
            if required_terms and not any(term in content for term in required_terms):
                continue
            score = sum(1 for term in terms if term in content)
            if score > 0:
                scored_matches.append((score, record))
        scored_matches.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored_matches[: max(1, limit)]]

    def list_recent(
        self,
        namespace: str = "default",
        limit: int = 5,
    ) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, namespace, type, content, created_at, updated_at
                FROM memories
                WHERE namespace = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (namespace, max(1, limit)),
            ).fetchall()
        return [_record_from_row(row) for row in rows]

    def update(
        self,
        memory_id: str,
        content: str,
        namespace: str = "default",
    ) -> MemoryRecord | None:
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, updated_at = ?
                WHERE id = ? AND namespace = ?
                """,
                (content.strip(), now, memory_id, namespace),
            )
            if cursor.rowcount == 0:
                return None
            row = conn.execute(
                """
                SELECT id, namespace, type, content, created_at, updated_at
                FROM memories
                WHERE id = ? AND namespace = ?
                """,
                (memory_id, namespace),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_namespace_updated
                ON memories(namespace, updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _search_terms(query: str) -> list[str]:
    normalized = query.lower().strip()
    if not normalized:
        return []
    ascii_terms = re.findall(r"[a-z0-9_-]{2,}", normalized)
    cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]+", normalized))
    cjk_terms = [
        cjk_text[index : index + 2]
        for index in range(max(0, len(cjk_text) - 1))
    ]
    terms = ascii_terms + cjk_terms
    deduped = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped


def _required_terms(terms: list[str]) -> list[str]:
    generic_ascii_terms = {"rag", "llm", "ai"}
    return [
        term
        for term in terms
        if re.fullmatch(r"[a-z0-9_-]{2,}", term) and term not in generic_ascii_terms
    ]


def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        namespace=row["namespace"],
        type=row["type"],
        content=row["content"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
