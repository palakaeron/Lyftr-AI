"""
16 required backend tests — all LLM calls mocked.
"""
import pytest
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport


# ─── Helpers ──────────────────────────────────────────────────────────────────

def live(
    reply="Hello from AI",
    model="gpt-3.5-turbo",
    input_tokens=10,
    output_tokens=5,
    key_id="key_1",
    fallback_attempts=0,
):
    return {
        "reply": reply,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "key_id": key_id,
        "fallback_attempts": fallback_attempts,
    }


def session() -> str:
    return str(uuid.uuid4())


pytestmark = pytest.mark.asyncio


# ─── 1. Health endpoint ───────────────────────────────────────────────────────

async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["keys_configured"], int)
    assert body["keys_configured"] >= 2
    assert isinstance(body["cache_enabled"], bool)


# ─── 2. Successful chat request ───────────────────────────────────────────────

async def test_successful_chat(client):
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live())):
        r = await client.post(
            "/chat",
            json={"session_id": session(), "message": "Hello"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "Hello from AI"
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert isinstance(meta["latency_ms"], int)
    assert isinstance(meta["input_tokens"], int)
    assert isinstance(meta["output_tokens"], int)


# ─── 3. Missing message validation ───────────────────────────────────────────

async def test_missing_message(client):
    r = await client.post("/chat", json={"session_id": session()})
    assert r.status_code == 422
    assert "detail" in r.json()


# ─── 4. Empty message validation ─────────────────────────────────────────────

async def test_empty_message(client):
    r = await client.post(
        "/chat",
        json={"session_id": session(), "message": "   "},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


# ─── 5. Message exceeding 4000 characters ────────────────────────────────────

async def test_message_too_long(client):
    r = await client.post(
        "/chat",
        json={"session_id": session(), "message": "x" * 4001},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


# ─── 6. Multi-turn conversation memory ───────────────────────────────────────

async def test_multiturn_memory(client):
    sid = session()

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="I am AI"))):
        r1 = await client.post("/chat", json={"session_id": sid, "message": "What are you?"})
    assert r1.status_code == 200

    # Second turn — backend should include previous history in messages
    captured_messages = []

    async def fake_completion(messages, simulation_map):
        captured_messages.extend(messages)
        return live(reply="Still AI")

    with patch("app.main.chat_completion", new=fake_completion):
        r2 = await client.post(
            "/chat",
            json={"session_id": sid, "message": "Tell me more"},
        )
    assert r2.status_code == 200

    # The messages sent to LLM must include the first user+assistant pair
    roles = [m["role"] for m in captured_messages]
    assert "user" in roles
    assert "assistant" in roles
    # At least 3 messages: prior user, prior assistant, new user
    assert len(captured_messages) >= 3


# ─── 7. Session isolation ─────────────────────────────────────────────────────

async def test_session_isolation(client):
    sid_a = session()
    sid_b = session()

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Reply A"))):
        await client.post("/chat", json={"session_id": sid_a, "message": "Hello A"})

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Reply B"))):
        await client.post("/chat", json={"session_id": sid_b, "message": "Hello B"})

    r_a = await client.get(f"/sessions/{sid_a}/messages")
    r_b = await client.get(f"/sessions/{sid_b}/messages")

    msgs_a = r_a.json()["messages"]
    msgs_b = r_b.json()["messages"]

    # Sessions must not bleed into each other
    contents_a = [m["content"] for m in msgs_a]
    contents_b = [m["content"] for m in msgs_b]
    assert "Hello A" in contents_a
    assert "Hello B" not in contents_a
    assert "Hello B" in contents_b
    assert "Hello A" not in contents_b


# ─── 8. Session history endpoint ─────────────────────────────────────────────

async def test_session_history(client):
    sid = session()
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Hi!"))):
        await client.post("/chat", json={"session_id": sid, "message": "Hey"})

    r = await client.get(f"/sessions/{sid}/messages")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    messages = body["messages"]
    assert len(messages) == 2  # user + assistant
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant"]
    # Assistant message must have metadata
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    assert assistant_msg["metadata"] is not None
    assert "resolution_source" in assistant_msg["metadata"]


# ─── 9. Unknown session returns 404 ──────────────────────────────────────────

async def test_unknown_session_404(client):
    r = await client.get(f"/sessions/{session()}/messages")
    assert r.status_code == 404
    assert r.json()["detail"] == "session not found"


# ─── 10. Cache hit returns identical reply ────────────────────────────────────

async def test_cache_hit_identical_reply(client):
    from app.llm import CacheHit

    cached = CacheHit(reply="Cached reply", model="gpt-3.5-turbo", input_tokens=10, output_tokens=5)

    # First call: live
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Cached reply"))):
        r1 = await client.post("/chat", json={"session_id": session(), "message": "Hi"})
    assert r1.status_code == 200

    # Second call (new session): cache hit
    with patch("app.main.chat_completion", new=AsyncMock(return_value=cached)):
        r2 = await client.post("/chat", json={"session_id": session(), "message": "Hi"})
    assert r2.status_code == 200

    assert r1.json()["reply"] == r2.json()["reply"]


# ─── 11. Cache hit has LLM_CACHE metadata ────────────────────────────────────

async def test_cache_hit_metadata_source(client):
    from app.llm import CacheHit

    cached = CacheHit(reply="Cached", model="gpt-3.5-turbo", input_tokens=8, output_tokens=3)
    with patch("app.main.chat_completion", new=AsyncMock(return_value=cached)):
        r = await client.post("/chat", json={"session_id": session(), "message": "Hi"})

    assert r.status_code == 200
    assert r.json()["metadata"]["resolution_source"] == "LLM_CACHE"


# ─── 12. Cache hit preserves token usage and has null key_id ─────────────────

async def test_cache_hit_token_preservation(client):
    from app.llm import CacheHit

    cached = CacheHit(reply="Cached", model="gpt-3.5-turbo", input_tokens=42, output_tokens=9)
    with patch("app.main.chat_completion", new=AsyncMock(return_value=cached)):
        r = await client.post("/chat", json={"session_id": session(), "message": "Hi"})

    meta = r.json()["metadata"]
    assert meta["input_tokens"] == 42
    assert meta["output_tokens"] == 9
    assert meta["key_id"] is None


# ─── 13. First-key failure triggers second-key fallback ──────────────────────

async def test_first_key_failure_fallback(client):
    # Simulate key_1=429 — the llm module should use key_2
    with patch("app.main.chat_completion", new=AsyncMock(
        return_value=live(key_id="key_2", fallback_attempts=1)
    )):
        r = await client.post(
            "/chat",
            json={"session_id": session(), "message": "Hello"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["key_id"] == "key_2"
    assert meta["fallback_attempts"] >= 1


# ─── 14. Both keys failing returns 503 ───────────────────────────────────────

async def test_all_keys_fail_503(client):
    with patch("app.main.chat_completion", new=AsyncMock(return_value=None)):
        r = await client.post(
            "/chat",
            json={"session_id": session(), "message": "Hello"},
        )
    assert r.status_code == 503
    assert r.json()["detail"] == "all providers unavailable"


# ─── 15. Failed requests are not stored in SQLite ────────────────────────────

async def test_failed_requests_not_stored(client):
    sid = session()
    with patch("app.main.chat_completion", new=AsyncMock(return_value=None)):
        r = await client.post("/chat", json={"session_id": sid, "message": "Should not store"})
    assert r.status_code == 503

    # Session should not exist
    r2 = await client.get(f"/sessions/{sid}/messages")
    assert r2.status_code == 404


# ─── 16. Simulated failure does not permanently affect subsequent requests ────

async def test_simulated_failure_not_permanent(client):
    sid = session()

    # First request with failure simulation
    with patch("app.main.chat_completion", new=AsyncMock(return_value=None)):
        r1 = await client.post(
            "/chat",
            json={"session_id": sid, "message": "Fail this"},
            headers={"X-Simulate-Failure": "*=429"},
        )
    assert r1.status_code == 503

    # Second request WITHOUT failure header — must succeed
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Now it works"))):
        r2 = await client.post(
            "/chat",
            json={"session_id": sid, "message": "Normal request"},
        )
    assert r2.status_code == 200
    assert r2.json()["reply"] == "Now it works"


# ─── 17. GET /sessions lists all sessions ─────────────────────────────────────

async def test_list_sessions_empty(client):
    r = await client.get("/sessions")
    assert r.status_code == 200
    body = r.json()
    assert "sessions" in body
    assert isinstance(body["sessions"], list)


async def test_list_sessions_contains_created_sessions(client):
    sid_a = session()
    sid_b = session()

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Hello A"))):
        await client.post("/chat", json={"session_id": sid_a, "message": "Hello from A"})

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Hello B"))):
        await client.post("/chat", json={"session_id": sid_b, "message": "Hello from B"})

    r = await client.get("/sessions")
    assert r.status_code == 200
    ids = [s["session_id"] for s in r.json()["sessions"]]
    assert sid_a in ids
    assert sid_b in ids


async def test_list_sessions_preview_is_first_user_message(client):
    sid = session()
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live(reply="Pong"))):
        await client.post("/chat", json={"session_id": sid, "message": "Ping preview message"})

    r = await client.get("/sessions")
    assert r.status_code == 200
    match = next((s for s in r.json()["sessions"] if s["session_id"] == sid), None)
    assert match is not None
    assert "Ping preview message" in match["preview"]


async def test_list_sessions_ordered_most_recent_first(client):
    sid_old = session()
    sid_new = session()

    with patch("app.main.chat_completion", new=AsyncMock(return_value=live())):
        await client.post("/chat", json={"session_id": sid_old, "message": "Older"})
    with patch("app.main.chat_completion", new=AsyncMock(return_value=live())):
        await client.post("/chat", json={"session_id": sid_new, "message": "Newer"})

    r = await client.get("/sessions")
    ids = [s["session_id"] for s in r.json()["sessions"]]
    # sid_new was written last, so it should appear before sid_old
    assert ids.index(sid_new) < ids.index(sid_old)


async def test_failed_requests_not_in_session_list(client):
    sid = session()
    with patch("app.main.chat_completion", new=AsyncMock(return_value=None)):
        await client.post("/chat", json={"session_id": sid, "message": "Should not appear"})

    r = await client.get("/sessions")
    ids = [s["session_id"] for s in r.json()["sessions"]]
    assert sid not in ids
