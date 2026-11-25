"""
Cross-session LLM cache tests.

Proves the exact cache-key behaviour described in the README:
  - Within one session the cache key always changes (history grows with every
    turn, so the complete message list is never the same twice).
  - Across two sessions the FIRST question produces an identical message list
    ([user: Q]) and therefore hits the cache on the second session.

All LLM calls are mocked at the app.main.chat_completion level so no real
API key or soundfile/audio machinery is needed.
"""
import pytest
import uuid
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


def _live(reply="4", input_tokens=10, output_tokens=2):
    return {
        "reply": reply,
        "model": "gpt-3.5-turbo",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "key_id": "key_1",
        "fallback_attempts": 0,
    }


def _cached(reply="4", input_tokens=10, output_tokens=2):
    from app.llm import CacheHit
    return CacheHit(reply=reply, model="gpt-3.5-turbo",
                    input_tokens=input_tokens, output_tokens=output_tokens)


def sid():
    return str(uuid.uuid4())


# ── Test 1: Session A, turn 1 → LIVE_INFERENCE ───────────────────────────────

async def test_first_request_is_live_inference(client):
    """The very first message in any session is always a live call."""
    with patch("app.main.chat_completion", new=AsyncMock(return_value=_live())):
        r = await client.post("/chat", json={"session_id": sid(), "message": "What is 2+2?"})
    assert r.status_code == 200
    assert r.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


# ── Test 2: Session A repeated question → LIVE_INFERENCE (history changed key) ─

async def test_same_session_repeated_question_is_live_inference(client):
    """
    Within one session, repeating the same question always produces a cache miss.

    Turn 1 LLM input:  [user: Q]
    Turn 2 LLM input:  [user: Q, assistant: A, user: Q]  <- different key

    The backend should call chat_completion twice, not once.
    """
    session_id = sid()
    messages_seen = []

    async def capture_and_live(messages, simulation_map):
        messages_seen.append(list(messages))  # snapshot
        return _live()

    # Turn 1
    with patch("app.main.chat_completion", new=capture_and_live):
        r1 = await client.post("/chat", json={"session_id": session_id, "message": "What is 2+2?"})
    assert r1.status_code == 200
    assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Turn 2 — same question
    with patch("app.main.chat_completion", new=capture_and_live):
        r2 = await client.post("/chat", json={"session_id": session_id, "message": "What is 2+2?"})
    assert r2.status_code == 200
    assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Verify that history caused a different input list
    assert len(messages_seen) == 2, "chat_completion should have been called twice"
    turn1_input = messages_seen[0]
    turn2_input = messages_seen[1]
    assert turn1_input != turn2_input, (
        "Turn 2 message list should differ from turn 1 because history is prepended"
    )
    assert len(turn1_input) == 1               # [user: Q]
    assert len(turn2_input) == 3               # [user: Q, assistant: A, user: Q]
    assert turn2_input[1]["role"] == "assistant"


# ── Test 3: Session B, same first question → LLM_CACHE ───────────────────────

async def test_cross_session_identical_first_question_is_cache_hit(client):
    """
    Two different sessions both sending the same message as their FIRST turn
    produce identical LLM inputs: [user: Q].
    The second session must return LLM_CACHE with key_id=null.
    """
    # Session A — live
    with patch("app.main.chat_completion", new=AsyncMock(return_value=_live(reply="4"))):
        r1 = await client.post("/chat", json={"session_id": sid(), "message": "What is 2+2?"})
    assert r1.status_code == 200
    assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # Session B — cache hit (backend returns a CacheHit object)
    with patch("app.main.chat_completion", new=AsyncMock(return_value=_cached(reply="4"))):
        r2 = await client.post("/chat", json={"session_id": sid(), "message": "What is 2+2?"})
    assert r2.status_code == 200
    meta2 = r2.json()["metadata"]
    assert meta2["resolution_source"] == "LLM_CACHE", (
        f"Cross-session identical first question should be LLM_CACHE, got {meta2['resolution_source']}"
    )
    assert meta2["key_id"] is None, "Cache hits must have key_id=null"
    assert meta2["input_tokens"] == 10
    assert meta2["output_tokens"] == 2


# ── Test 4: Cache key analysis — verify message payloads ─────────────────────

async def test_message_payload_differs_between_session_turns(client):
    """
    Directly capture and compare the full message payloads sent to the LLM
    for the two turns within one session and confirm:
      turn 1: [user: Q]
      turn 2: [user: Q, assistant: A, user: Q]
    """
    session_id = sid()
    captured = []

    async def capture(messages, simulation_map):
        captured.append([dict(m) for m in messages])
        return _live(reply="The answer is 4")

    with patch("app.main.chat_completion", new=capture):
        await client.post("/chat", json={"session_id": session_id, "message": "What is 2+2?"})
    with patch("app.main.chat_completion", new=capture):
        await client.post("/chat", json={"session_id": session_id, "message": "What is 2+2?"})

    t1 = captured[0]
    t2 = captured[1]

    # Turn 1: exactly [user: Q]
    assert t1 == [{"role": "user", "content": "What is 2+2?"}]

    # Turn 2: [user: Q, assistant: A, user: Q]
    assert t2[0] == {"role": "user",      "content": "What is 2+2?"}
    assert t2[1] == {"role": "assistant", "content": "The answer is 4"}
    assert t2[2] == {"role": "user",      "content": "What is 2+2?"}

    # Prove the cache keys would differ
    import litellm
    from litellm import Cache
    orig = litellm.cache
    litellm.cache = Cache(type="local")
    litellm.enable_cache()
    try:
        key1 = litellm.cache.get_cache_key(model="gpt-3.5-turbo", messages=t1)
        key2 = litellm.cache.get_cache_key(model="gpt-3.5-turbo", messages=t2)
        assert key1 != key2, "Same question in same session must produce different cache keys due to history"
    finally:
        litellm.cache = orig


# ── Test 5: Cross-session cache key identity ──────────────────────────────────

def test_cross_session_first_question_produces_identical_cache_key():
    """
    Unit test: two fresh sessions asking the same first question produce
    exactly the same LiteLLM cache key (no session ID in the key, only messages+model).
    """
    import litellm
    from litellm import Cache

    orig = litellm.cache
    litellm.cache = Cache(type="local")
    litellm.enable_cache()
    try:
        msgs_a = [{"role": "user", "content": "What is 2+2?"}]
        msgs_b = [{"role": "user", "content": "What is 2+2?"}]  # identical

        key_a = litellm.cache.get_cache_key(model="groq/openai/gpt-oss-120b", messages=msgs_a)
        key_b = litellm.cache.get_cache_key(model="groq/openai/gpt-oss-120b", messages=msgs_b)

        assert key_a == key_b, (
            f"Identical first-turn message lists must produce the same cache key.\n"
            f"  key_a: {key_a}\n  key_b: {key_b}"
        )
    finally:
        litellm.cache = orig


def test_different_api_keys_produce_same_cache_key():
    """
    api_key is NOT part of the LiteLLM cache key.
    Two calls with different API keys but identical messages share the same cache.
    """
    import litellm
    from litellm import Cache

    orig = litellm.cache
    litellm.cache = Cache(type="local")
    litellm.enable_cache()
    try:
        msgs = [{"role": "user", "content": "What is 2+2?"}]

        key1 = litellm.cache.get_cache_key(model="groq/openai/gpt-oss-120b", messages=msgs, api_key="gsk_key_one")
        key2 = litellm.cache.get_cache_key(model="groq/openai/gpt-oss-120b", messages=msgs, api_key="gsk_key_two")

        assert key1 == key2, (
            "api_key must NOT be part of the cache key — different keys for the same input must share cached results"
        )
    finally:
        litellm.cache = orig
