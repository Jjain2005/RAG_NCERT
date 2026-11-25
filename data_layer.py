"""Lightweight SQLite persistence layer for Chainlit chat history."""
import asyncio
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from chainlit.data.base import BaseDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.types import (
    Feedback,
    FeedbackDict,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User

if TYPE_CHECKING:
    from chainlit.element import Element, ElementDict
    from chainlit.step import StepDict

DB_PATH = Path(__file__).resolve().parent / "data" / "chat_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    identifier TEXT UNIQUE NOT NULL,
    createdAt TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    createdAt TEXT NOT NULL,
    name TEXT,
    userId TEXT,
    userIdentifier TEXT,
    tags TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    threadId TEXT NOT NULL,
    parentId TEXT,
    streaming INTEGER NOT NULL DEFAULT 0,
    waitForAnswer INTEGER,
    isError INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT,
    input TEXT DEFAULT '',
    output TEXT DEFAULT '',
    createdAt TEXT,
    start TEXT,
    "end" TEXT,
    generation TEXT,
    showInput TEXT,
    language TEXT
);

CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps(threadId, createdAt);

CREATE TABLE IF NOT EXISTS elements (
    id TEXT PRIMARY KEY,
    threadId TEXT,
    type TEXT NOT NULL,
    chainlitKey TEXT,
    url TEXT,
    objectKey TEXT,
    name TEXT NOT NULL,
    display TEXT NOT NULL,
    size TEXT,
    language TEXT,
    page INTEGER,
    autoPlay INTEGER,
    playerConfig TEXT,
    forId TEXT,
    mime TEXT,
    props TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS feedbacks (
    id TEXT PRIMARY KEY,
    forId TEXT NOT NULL,
    threadId TEXT,
    value INTEGER NOT NULL,
    comment TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_load(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return raw


class SQLiteDataLayer(BaseDataLayer):
    """Thread-safe SQLite implementation of Chainlit's BaseDataLayer."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def _run(self, fn, *args, **kwargs):
        return asyncio.to_thread(self._locked_call, fn, *args, **kwargs)

    def _locked_call(self, fn, *args, **kwargs):
        with self._lock:
            return fn(*args, **kwargs)

    # ------ Users ------

    def _get_user_sync(self, identifier: str) -> Optional[PersistedUser]:
        row = self._conn.execute(
            "SELECT * FROM users WHERE identifier = ?", (identifier,)
        ).fetchone()
        if not row:
            return None
        return PersistedUser(
            id=row["id"],
            identifier=row["identifier"],
            createdAt=row["createdAt"],
            metadata=_json_load(row["metadata"], {}),
        )

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return await self._run(self._get_user_sync, identifier)

    def _create_user_sync(self, user: User) -> Optional[PersistedUser]:
        existing = self._get_user_sync(user.identifier)
        meta_json = json.dumps(user.metadata or {})
        if not existing:
            self._conn.execute(
                "INSERT INTO users (id, identifier, createdAt, metadata) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), str(user.identifier), _now_iso(), meta_json),
            )
        else:
            self._conn.execute(
                "UPDATE users SET metadata = ? WHERE identifier = ?",
                (meta_json, str(user.identifier)),
            )
        self._conn.commit()
        return self._get_user_sync(user.identifier)

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        return await self._run(self._create_user_sync, user)

    # ------ Threads ------

    def _get_thread_author_sync(self, thread_id: str) -> str:
        row = self._conn.execute(
            "SELECT userIdentifier FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row and row["userIdentifier"] is not None:
            return row["userIdentifier"]
        raise ValueError(f"Author not found for thread_id {thread_id}")

    async def get_thread_author(self, thread_id: str) -> str:
        return await self._run(self._get_thread_author_sync, thread_id)

    def _update_thread_sync(
        self,
        thread_id: str,
        name: Optional[str],
        user_id: Optional[str],
        metadata: Optional[Dict],
        tags: Optional[List[str]],
    ):
        user_identifier = None
        if user_id:
            urow = self._conn.execute(
                "SELECT identifier FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if urow:
                user_identifier = urow["identifier"]

        existing = self._conn.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()

        has_updates = any(x is not None for x in (metadata, name, user_id, tags))
        if existing and not has_updates:
            return

        base_meta = _json_load(existing["metadata"], {}) if existing else {}
        if metadata:
            for k, v in metadata.items():
                if v is None:
                    base_meta.pop(k, None)
                else:
                    base_meta[k] = v

        name_val = name if name is not None else base_meta.get("name")
        if existing:
            self._conn.execute(
                """
                UPDATE threads SET
                    name = COALESCE(?, name),
                    userId = COALESCE(?, userId),
                    userIdentifier = COALESCE(?, userIdentifier),
                    tags = COALESCE(?, tags),
                    metadata = ?
                WHERE id = ?
                """,
                (
                    name_val,
                    user_id,
                    user_identifier,
                    json.dumps(tags) if tags is not None else None,
                    json.dumps(base_meta),
                    thread_id,
                ),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO threads (id, createdAt, name, userId, userIdentifier, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    _now_iso(),
                    name_val,
                    user_id,
                    user_identifier,
                    json.dumps(tags) if tags is not None else None,
                    json.dumps(base_meta),
                ),
            )
        self._conn.commit()

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        await self._run(
            self._update_thread_sync, thread_id, name, user_id, metadata, tags
        )

    def _delete_thread_sync(self, thread_id: str):
        self._conn.execute(
            "DELETE FROM feedbacks WHERE forId IN (SELECT id FROM steps WHERE threadId = ?)",
            (thread_id,),
        )
        self._conn.execute("DELETE FROM elements WHERE threadId = ?", (thread_id,))
        self._conn.execute("DELETE FROM steps WHERE threadId = ?", (thread_id,))
        self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self._conn.commit()

    async def delete_thread(self, thread_id: str):
        await self._run(self._delete_thread_sync, thread_id)

    def _row_to_step(self, row: sqlite3.Row) -> "StepDict":
        feedback = None
        if row["feedback_value"] is not None:
            feedback = FeedbackDict(
                forId=row["id"],
                id=row["feedback_id"],
                value=row["feedback_value"],
                comment=row["feedback_comment"],
            )
        show_input = row["showInput"]
        return {
            "id": row["id"],
            "name": row["name"],
            "type": row["type"],
            "threadId": row["threadId"],
            "parentId": row["parentId"],
            "streaming": bool(row["streaming"]),
            "waitForAnswer": (
                bool(row["waitForAnswer"])
                if row["waitForAnswer"] is not None
                else None
            ),
            "isError": bool(row["isError"]) if row["isError"] is not None else None,
            "metadata": _json_load(row["metadata"], {}),
            "tags": _json_load(row["tags"], None),
            "input": (row["input"] or "") if show_input not in [None, "false"] else "",
            "output": row["output"] or "",
            "createdAt": row["createdAt"],
            "start": row["start"],
            "end": row["end"],
            "generation": _json_load(row["generation"], None),
            "showInput": show_input,
            "language": row["language"],
            "feedback": feedback,
        }

    def _get_threads_sync(
        self, user_id: Optional[str] = None, thread_id: Optional[str] = None
    ) -> List[ThreadDict]:
        where = "t.userId = ?" if user_id else "t.id = ?"
        param = user_id if user_id else thread_id
        thread_rows = self._conn.execute(
            f"""
            SELECT t.*, COALESCE(MAX(s.createdAt), t.createdAt) AS updatedAt
            FROM threads t
            LEFT JOIN steps s ON t.id = s.threadId
            WHERE {where}
            GROUP BY t.id
            ORDER BY updatedAt DESC
            """,
            (param,),
        ).fetchall()

        if not thread_rows:
            return []

        threads: Dict[str, ThreadDict] = {}
        for row in thread_rows:
            threads[row["id"]] = ThreadDict(
                id=row["id"],
                createdAt=row["createdAt"],
                name=row["name"],
                userId=row["userId"],
                userIdentifier=row["userIdentifier"],
                tags=_json_load(row["tags"], None),
                metadata=_json_load(row["metadata"], {}),
                steps=[],
                elements=[],
            )

        placeholders = ",".join("?" for _ in threads)
        tids = list(threads.keys())

        step_rows = self._conn.execute(
            f"""
            SELECT s.*, f.value AS feedback_value, f.comment AS feedback_comment, f.id AS feedback_id
            FROM steps s
            LEFT JOIN feedbacks f ON s.id = f.forId
            WHERE s.threadId IN ({placeholders})
            ORDER BY s.createdAt ASC
            """,
            tids,
        ).fetchall()
        for srow in step_rows:
            threads[srow["threadId"]]["steps"].append(self._row_to_step(srow))

        elem_rows = self._conn.execute(
            f"SELECT * FROM elements WHERE threadId IN ({placeholders})", tids
        ).fetchall()
        for erow in elem_rows:
            threads[erow["threadId"]]["elements"].append(
                {
                    "id": erow["id"],
                    "threadId": erow["threadId"],
                    "type": erow["type"],
                    "chainlitKey": erow["chainlitKey"],
                    "url": erow["url"],
                    "objectKey": erow["objectKey"],
                    "name": erow["name"],
                    "display": erow["display"],
                    "size": erow["size"],
                    "language": erow["language"],
                    "autoPlay": (
                        bool(erow["autoPlay"])
                        if erow["autoPlay"] is not None
                        else None
                    ),
                    "playerConfig": _json_load(erow["playerConfig"], None),
                    "page": erow["page"],
                    "props": _json_load(erow["props"], {}),
                    "forId": erow["forId"],
                    "mime": erow["mime"],
                }
            )

        return list(threads.values())

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        rows = await self._run(self._get_threads_sync, None, thread_id)
        return rows[0] if rows else None

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        if not filters.userId:
            raise ValueError("userId is required")
        all_threads = await self._run(self._get_threads_sync, filters.userId, None)

        search_kw = filters.search.lower() if filters.search else None
        fb_val = int(filters.feedback) if filters.feedback is not None else None

        filtered = []
        for t in all_threads:
            if search_kw:
                in_name = search_kw in (t.get("name") or "").lower()
                in_steps = any(
                    search_kw in (s.get("output") or "").lower()
                    or search_kw in (s.get("input") or "").lower()
                    for s in t["steps"]
                )
                if not (in_name or in_steps):
                    continue
            if fb_val is not None:
                if not any(
                    (s.get("feedback") or {}).get("value") == fb_val
                    for s in t["steps"]
                ):
                    continue
            filtered.append(t)

        start = 0
        if pagination.cursor:
            for i, t in enumerate(filtered):
                if t["id"] == pagination.cursor:
                    start = i + 1
                    break
        end = start + pagination.first
        page = filtered[start:end]

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=len(filtered) > end,
                startCursor=page[0]["id"] if page else None,
                endCursor=page[-1]["id"] if page else None,
            ),
            data=page,
        )

    # ------ Steps ------

    def _upsert_step_sync(self, step_dict: "StepDict"):
        self._update_thread_sync(step_dict["threadId"], None, None, None, None)

        show_input = step_dict.get("showInput")
        show_input_str = str(show_input).lower() if show_input is not None else None

        self._conn.execute(
            """
            INSERT INTO steps (
                id, name, type, threadId, parentId, streaming, waitForAnswer,
                isError, metadata, tags, input, output, createdAt, start, "end",
                generation, showInput, language
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                parentId = excluded.parentId,
                streaming = excluded.streaming,
                waitForAnswer = excluded.waitForAnswer,
                isError = excluded.isError,
                metadata = excluded.metadata,
                tags = excluded.tags,
                input = excluded.input,
                output = excluded.output,
                start = excluded.start,
                "end" = excluded."end",
                generation = excluded.generation,
                showInput = excluded.showInput,
                language = excluded.language
            """,
            (
                step_dict["id"],
                step_dict.get("name", ""),
                step_dict.get("type", "undefined"),
                step_dict["threadId"],
                step_dict.get("parentId"),
                int(bool(step_dict.get("streaming", False))),
                (
                    int(bool(step_dict["waitForAnswer"]))
                    if step_dict.get("waitForAnswer") is not None
                    else None
                ),
                (
                    int(bool(step_dict["isError"]))
                    if step_dict.get("isError") is not None
                    else None
                ),
                json.dumps(step_dict.get("metadata") or {}),
                (
                    json.dumps(step_dict["tags"])
                    if step_dict.get("tags") is not None
                    else None
                ),
                step_dict.get("input", ""),
                step_dict.get("output", ""),
                step_dict.get("createdAt") or _now_iso(),
                step_dict.get("start"),
                step_dict.get("end"),
                json.dumps(step_dict.get("generation") or {}),
                show_input_str,
                step_dict.get("language"),
            ),
        )
        self._conn.commit()

    @queue_until_user_message()
    async def create_step(self, step_dict: "StepDict"):
        await self._run(self._upsert_step_sync, step_dict)

    @queue_until_user_message()
    async def update_step(self, step_dict: "StepDict"):
        await self._run(self._upsert_step_sync, step_dict)

    def _delete_step_sync(self, step_id: str):
        self._conn.execute("DELETE FROM feedbacks WHERE forId = ?", (step_id,))
        self._conn.execute("DELETE FROM elements WHERE forId = ?", (step_id,))
        self._conn.execute("DELETE FROM steps WHERE id = ?", (step_id,))
        self._conn.commit()

    @queue_until_user_message()
    async def delete_step(self, step_id: str):
        await self._run(self._delete_step_sync, step_id)

    # ------ Feedback ------

    def _upsert_feedback_sync(self, feedback: Feedback) -> str:
        feedback.id = feedback.id or str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO feedbacks (id, forId, threadId, value, comment)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                value = excluded.value,
                comment = excluded.comment
            """,
            (
                feedback.id,
                feedback.forId,
                feedback.threadId,
                feedback.value,
                feedback.comment,
            ),
        )
        self._conn.commit()
        return feedback.id

    async def upsert_feedback(self, feedback: Feedback) -> str:
        return await self._run(self._upsert_feedback_sync, feedback)

    def _delete_feedback_sync(self, feedback_id: str) -> bool:
        self._conn.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
        self._conn.commit()
        return True

    async def delete_feedback(self, feedback_id: str) -> bool:
        return await self._run(self._delete_feedback_sync, feedback_id)

    # ------ Elements ------

    @queue_until_user_message()
    async def create_element(self, element: "Element"):
        pass

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional["ElementDict"]:
        return None

    @queue_until_user_message()
    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    # ------ Favorites & Lifecycle ------

    def _get_favorite_steps_sync(self, user_id: str) -> List["StepDict"]:
        rows = self._conn.execute(
            """
            SELECT s.*, NULL AS feedback_value, NULL AS feedback_comment, NULL AS feedback_id
            FROM steps s
            JOIN threads t ON s.threadId = t.id
            WHERE t.userId = ? AND s.metadata LIKE '%"favorite": true%'
            ORDER BY s.createdAt DESC
            """,
            (user_id,),
        ).fetchall()
        return [
            self._row_to_step(r)
            for r in rows
            if _json_load(r["metadata"], {}).get("favorite")
        ]

    async def get_favorite_steps(self, user_id: str) -> List["StepDict"]:
        return await self._run(self._get_favorite_steps_sync, user_id)

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        with self._lock:
            self._conn.close()

