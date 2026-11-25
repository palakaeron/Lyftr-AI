"""
Pytest configuration and shared fixtures.
All LLM calls are mocked — no real API credits consumed.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Point to a temp DB before importing the app
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_db.name}")
os.environ.setdefault("LLM_API_KEYS", "test_key_1,test_key_2")
os.environ.setdefault("LLM_MODEL", "gpt-3.5-turbo")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
os.environ.setdefault("ENABLE_FAILURE_SIMULATION", "true")
os.environ.setdefault("CACHE_ENABLED", "true")
os.environ.setdefault("CACHE_TYPE", "in-memory")

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from httpx import AsyncClient, ASGITransport


def make_live_response(
    reply: str = "Hello from AI",
    model: str = "gpt-3.5-turbo",
    input_tokens: int = 10,
    output_tokens: int = 5,
    key_id: str = "key_1",
    fallback_attempts: int = 0,
) -> dict:
    return {
        "reply": reply,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "key_id": key_id,
        "fallback_attempts": fallback_attempts,
    }


@pytest.fixture(scope="function", autouse=True)
def reset_db():
    """Wipe and re-create the test database for each test."""
    from app.database import init_db, DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS messages")
    conn.commit()
    conn.close()
    init_db()
    yield


@pytest.fixture
def app():
    from app.main import app as _app
    return _app


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
