"""
SQLite persistence layer.
"""
import os
import json
import sqlite3
import logging
from typing import Optional, List

from .config import DB_PATH

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the database directory and tables on startup."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                metadata    TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id)"
        )
        conn.commit()
        logger.info("Database ready at %s", DB_PATH)
    finally:
        conn.close()


def check_db() -> bool:
    try:
        conn = _get_connection()
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


def get_db():
    """Dependency — not used directly but useful for extensions."""
    conn = _get_connection()
    try:
        yield conn
    finally:
        conn.close()


def save_message(
    session_id: str, role: str, content: str, metadata: Optional[dict]
) -> None:
    meta_json = json.dumps(metadata) if metadata else None
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, meta_json),
        )
        conn.commit()
    finally:
        conn.close()


def get_messages(session_id: str) -> Optional[List[dict]]:
    """
    Returns list of messages for the session, or None if the session does not exist.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        # Distinguish between "never existed" and "exists but empty"
        # We treat no rows as session-not-found for GET endpoint, but that's
        # handled in the route. Here we return None to signal no messages.
        return None

    result = []
    for row in rows:
        meta = None
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except json.JSONDecodeError:
                meta = None
        result.append({"role": row["role"], "content": row["content"], "metadata": meta})
    return result
