"""
Controlled HTTP-level cache trace test.
Proves the root cause of LIVE_INFERENCE on same-session repeats.
Runs through the actual FastAPI app with real LiteLLM cache pipeline.
"""
import asyncio
import os
import sys
import logging
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///./data/trace_test.db")
os.environ.setdefault("LLM_API_KEYS", "sk-trace,sk-trace-2")
os.environ.setdefault("LLM_MODEL", "gpt-3.5-turbo")
os.environ.setdefault("CACHE_ENABLED", "true")
os.environ.setdefault("CACHE_TYPE", "in-memory")
os.environ.setdefault("ENABLE_FAILURE_SIMULATION", "false")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

for name in ["LiteLLM", "LiteLLM Proxy", "app.llm", "app.main", "app.database"]:
    logging.getLogger(name).setLevel(logging.ERROR)

import litellm
from litellm import Cache
from litellm.utils import ModelResponse
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

PATCH = "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion"


def _response(content="42"):
    return ModelResponse(
        id="x",
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop", "index": 0}],
        model="gpt-3.5-turbo",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )


@pytest.fixture(autouse=True)
def fresh_state():
    """Fresh DB and fresh in-memory cache for every test."""
    from app.database import init_db, DB_PATH
    import sqlite3

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS messages")
    conn.commit()
    conn.close()
    init_db()

    litellm.cache = Cache(type="local")
    litellm.enable_cache()
    yield
    litellm.cache = None


@pytest.fixture
async def client():
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _post(client, session_id, message, counter):
    """POST /chat and yield until cache write tasks complete."""
    with patch(PATCH, new=AsyncMock(side_effect=lambda *a, **kw: (counter.__setitem__("n", counter["n"] + 1), _response())[1])):
        r = await client.post("/chat", json={"session_id": session_id, "message": message})
    for _ in range(40):
        await asyncio.sleep(0)
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Same session, same question twice → ALWAYS two provider calls
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_same_session_same_question_is_never_cached(client):
    """
    Root cause test: within a single session, sending the same question twice
    always produces different LLM message lists because history grows.

    Turn 1 sends:  [user: Q]
    Turn 2 sends:  [user: Q, assistant: A, user: Q]  ← different cache key
    """
    counter = {"n": 0}

    r1 = await _post(client, "sess-repeat", "What is 6x7?", counter)
    calls_after_1 = counter["n"]

    r2 = await _post(client, "sess-repeat", "What is 6x7?", counter)
    calls_after_2 = counter["n"]

    # Both calls are LIVE_INFERENCE because history changes the message list
    assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r2["metadata"]["resolution_source"] == "LIVE_INFERENCE", (
        "Second identical question in same session should still be LIVE_INFERENCE "
        "because conversation history is prepended, making it a different cache key"
    )
    assert calls_after_1 == 1
    assert calls_after_2 == 2, (
        f"Expected 2 total provider calls (one per turn), got {calls_after_2}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Cross-session identical FIRST question → cache hit on second session
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_cross_session_identical_first_question_hits_cache(client):
    """
    Two different sessions both sending the same message as their first turn
    produce identical message lists: [user: Q]
    The second session should be served from cache with zero provider calls.
    """
    counter = {"n": 0}

    r1 = await _post(client, "cross-A", "What is 6x7?", counter)
    calls_after_1 = counter["n"]

    r2 = await _post(client, "cross-B", "What is 6x7?", counter)
    calls_after_2 = counter["n"]

    assert r1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert r2["metadata"]["resolution_source"] == "LLM_CACHE", (
        f"Second session with identical first question should be LLM_CACHE, "
        f"got {r2['metadata']['resolution_source']}"
    )
    assert calls_after_1 == 1
    assert calls_after_2 == 1, (
        f"Second cross-session call should reuse cache (0 extra provider calls). "
        f"Total calls: {calls_after_2}"
    )
    # Cache hit preserves original tokens and sets key_id=null
    assert r2["metadata"]["key_id"] is None
    assert r2["metadata"]["input_tokens"] == 10
    assert r2["metadata"]["output_tokens"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Cross-session identical SECOND turn does NOT falsely hit cache
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_cross_session_second_turn_no_false_cache_hit(client):
    """
    Session A has 2 turns. Session B sends the same second-turn question
    but as its FIRST turn. The message lists are different:
      A turn 2: [user: Q1, assistant: A1, user: Q2]
      B turn 1: [user: Q2]
    These must NOT produce a cache hit for B.
    """
    counter = {"n": 0}

    # Session A: two turns
    await _post(client, "ctx-A", "Who invented the telephone?", counter)
    await _post(client, "ctx-A", "What year was that?", counter)
    calls_after_a = counter["n"]

    # Session B: asks the same second question but as its first turn
    r_b = await _post(client, "ctx-B", "What year was that?", counter)
    calls_after_b = counter["n"]

    assert r_b["metadata"]["resolution_source"] == "LIVE_INFERENCE", (
        "Session B's first question should NOT hit the cache from Session A's "
        "second turn — the full message lists differ"
    )
    assert calls_after_b == calls_after_a + 1
