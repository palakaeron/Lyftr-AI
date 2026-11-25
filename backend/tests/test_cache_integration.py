"""
Controlled integration test for LiteLLM caching.

Proves:
  1. First call makes exactly ONE provider call; result stored in cache.
  2. Second call with IDENTICAL messages makes ZERO provider calls;
     litellm sets _hidden_params["cache_hit"] = True.
  3. Second call returns the same reply and tokens as the first.
  4. A DIFFERENT prompt makes a second provider call (no false cache hit).
  5. cache_hit detection in chat_completion() returns a CacheHit dataclass.

No real API key is used. The OpenAI-compatible provider call is mocked at
the HTTP transport level (openai AsyncCompletions.create), so litellm's
full async wrapper pipeline — including cache.get and cache.set — executes
unchanged.
"""
import asyncio
import os
import sys

import pytest

# ── env must be set before any app import ─────────────────────────────────────
os.environ.setdefault("CACHE_ENABLED", "true")
os.environ.setdefault("CACHE_TYPE", "in-memory")
os.environ.setdefault("LLM_API_KEYS", "sk-test-integration,sk-test-integration-2")
os.environ.setdefault("LLM_MODEL", "gpt-3.5-turbo")
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/test_integration.db")
os.environ.setdefault("ENABLE_FAILURE_SIMULATION", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import litellm
from litellm import Cache
from litellm.utils import ModelResponse
from unittest.mock import AsyncMock, MagicMock, patch

# ── Reset to a fresh in-memory cache for every test module run ────────────────
litellm.cache = Cache(type="local")
litellm.enable_cache()
litellm.set_verbose = False


# ── Helper: build a valid ModelResponse that litellm accepts ─────────────────

def _mock_response(content: str = "Paris is the capital of France.") -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        model="gpt-3.5-turbo",
        usage={"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
    )


# ── Fixture: fresh in-memory cache per test ───────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_cache():
    litellm.cache = Cache(type="local")
    litellm.enable_cache()
    yield
    # wipe after test
    litellm.cache = None


# ── Core helper: call litellm.acompletion with mock provider, yield for write ─

async def _call(messages, mock_fn) -> ModelResponse:
    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion",
        new=AsyncMock(side_effect=mock_fn),
    ):
        resp = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=messages,
            api_key="sk-test-integration",
            caching=True,
        )
    # Yield so the async cache write task (asyncio.create_task) completes
    for _ in range(20):
        await asyncio.sleep(0)
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — First call: provider is invoked, response stored; no cache_hit flag
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_first_call_is_live():
    call_count = 0

    def provider(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response()

    msgs = [{"role": "user", "content": "Capital of France?"}]
    resp = await _call(msgs, provider)

    assert call_count == 1, f"Expected 1 provider call, got {call_count}"
    assert resp.choices[0].message.content == "Paris is the capital of France."
    assert resp._hidden_params.get("cache_hit") is not True, (
        "First call should NOT be a cache hit"
    )

    # Cache should now contain one entry
    assert len(litellm.cache.cache.cache_dict) == 1, "Cache should have 1 entry after first call"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Second identical call: ZERO provider calls; cache_hit=True in _hidden_params
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_second_identical_call_is_cache_hit():
    call_count = 0

    def provider(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response()

    msgs = [{"role": "user", "content": "Capital of France?"}]

    # Warm the cache
    await _call(msgs, provider)
    assert call_count == 1

    # Second call — provider must NOT be called
    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion",
        new=AsyncMock(side_effect=provider),
    ):
        resp2 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=msgs,
            api_key="sk-test-integration",
            caching=True,
        )

    assert call_count == 1, (
        f"Provider was called {call_count} times total; expected exactly 1 "
        "(second call should have been served from cache)"
    )
    assert resp2._hidden_params.get("cache_hit") is True, (
        f"_hidden_params['cache_hit'] should be True on second call, "
        f"got: {resp2._hidden_params}"
    )
    assert resp2.choices[0].message.content == "Paris is the capital of France."


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Tokens and reply are preserved from the original response
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_cache_hit_preserves_tokens_and_reply():
    call_count = 0

    def provider(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response("Cached answer")

    msgs = [{"role": "user", "content": "Tell me something"}]
    resp1 = await _call(msgs, provider)

    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion",
        new=AsyncMock(side_effect=provider),
    ):
        resp2 = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=msgs,
            api_key="sk-test-integration",
            caching=True,
        )

    assert call_count == 1
    assert resp2.choices[0].message.content == resp1.choices[0].message.content
    assert resp2.usage.prompt_tokens == resp1.usage.prompt_tokens
    assert resp2.usage.completion_tokens == resp1.usage.completion_tokens


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Different prompt does NOT hit cache; makes a new provider call
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_different_prompt_is_not_cached():
    call_count = 0

    def provider(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response(f"Response {call_count}")

    msgs_a = [{"role": "user", "content": "Capital of France?"}]
    msgs_b = [{"role": "user", "content": "Capital of Germany?"}]

    await _call(msgs_a, provider)
    assert call_count == 1

    resp_b = await _call(msgs_b, provider)
    assert call_count == 2, (
        f"Different prompt should have triggered a second provider call, "
        f"but total calls = {call_count}"
    )
    assert resp_b._hidden_params.get("cache_hit") is not True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — chat_completion() in app.llm returns CacheHit on second call
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_app_llm_returns_cache_hit_dataclass():
    from app.llm import chat_completion, CacheHit

    # Temporarily inject a fresh key pool so chat_completion has a key
    import app.llm as llm_module
    orig_pool = llm_module.KEY_POOL.copy()
    llm_module.KEY_POOL = {"key_1": "sk-test-integration"}

    call_count = 0

    def provider(*a, **kw):
        nonlocal call_count
        call_count += 1
        return _mock_response("42 is the answer")

    msgs = [{"role": "user", "content": "What is the answer?"}]

    try:
        # Call 1 — live
        result1 = await _call_via_app(msgs, provider, chat_completion)
        assert not isinstance(result1, CacheHit), "First call should be LIVE_INFERENCE"
        assert result1["reply"] == "42 is the answer"
        assert call_count == 1

        # Call 2 — should be CacheHit
        result2 = await _call_via_app(msgs, provider, chat_completion)
        assert isinstance(result2, CacheHit), (
            f"Second identical call should return CacheHit, got {type(result2)}: {result2}"
        )
        assert result2.reply == "42 is the answer"
        assert result2.input_tokens == 12
        assert result2.output_tokens == 7
        assert call_count == 1  # still 1 — no second provider call
    finally:
        llm_module.KEY_POOL = orig_pool


async def _call_via_app(messages, mock_fn, chat_completion_fn) -> object:
    """Call app.llm.chat_completion with mocked provider, yield for cache write."""
    with patch(
        "litellm.llms.openai.openai.OpenAIChatCompletion.acompletion",
        new=AsyncMock(side_effect=mock_fn),
    ):
        result = await chat_completion_fn(messages, {})
    for _ in range(20):
        await asyncio.sleep(0)
    return result
