"""
LiteLLM integration: key pool, automatic failover, and response caching.

Uses acompletion() (async) so the async wrapper in litellm's utils.py
runs — that wrapper is what invokes cache.get / cache.set internally.
"""
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import litellm
from litellm import acompletion, Cache
from litellm.exceptions import (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    Timeout,
)

from .config import (
    LLM_API_KEYS,
    LLM_MODEL,
    CACHE_ENABLED,
    CACHE_TYPE,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)

# ── Key pool ───────────────────────────────────────────────────────────────────
KEY_POOL: Dict[str, str] = {}
for idx, key in enumerate(LLM_API_KEYS, start=1):
    KEY_POOL[f"key_{idx}"] = key

# ── Cache setup ────────────────────────────────────────────────────────────────
if CACHE_ENABLED:
    if CACHE_TYPE == "disk":
        os.makedirs(CACHE_DIR, exist_ok=True)
        litellm.cache = Cache(type="disk", disk_cache_dir=CACHE_DIR)
    else:
        litellm.cache = Cache(type="local")
    litellm.enable_cache()
    logger.info("LiteLLM cache enabled (%s)", CACHE_TYPE)
else:
    litellm.disable_cache()
    logger.info("LiteLLM cache disabled")

litellm.set_verbose = False


@dataclass
class CacheHit:
    """Returned when the response came from LiteLLM cache."""
    reply: str
    model: str
    input_tokens: int
    output_tokens: int


# ── Retriable errors ───────────────────────────────────────────────────────────
_RETRIABLE = (
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout,
)

_RETRIABLE_STATUS = {429, 500, 502, 503}


def _is_retriable(exc: Exception) -> bool:
    if isinstance(exc, _RETRIABLE):
        return True
    status = getattr(exc, "status_code", None)
    if status in _RETRIABLE_STATUS:
        return True
    return False


# ── Main chat function ─────────────────────────────────────────────────────────

async def chat_completion(
    messages: List[Dict[str, str]],
    simulation_map: Dict[str, int],
) -> Optional[Any]:
    """
    Attempt completion with each key in order.

    Uses acompletion() (async variant) so LiteLLM's async wrapper handles
    cache.get (before the provider call) and cache.set (after). The sync
    completion() path has its own wrapper but calling it inside async code
    can deadlock; the async path is correct here.

    Returns:
      - CacheHit  if LiteLLM served from cache
      - dict      with live inference results
      - None      if all keys failed
    """
    if not KEY_POOL:
        logger.error("No API keys configured")
        return None

    fallback_attempts = 0

    for key_id, api_key in KEY_POOL.items():
        # ── Failure simulation ─────────────────────────────────────────────
        sim_code = simulation_map.get(key_id) or simulation_map.get("*")
        if sim_code is not None:
            logger.info("Simulating failure %s for %s", sim_code, key_id)
            fallback_attempts += 1
            continue

        # ── LiteLLM async call ─────────────────────────────────────────────
        try:
            response = await acompletion(
                model=LLM_MODEL,
                messages=messages,
                api_key=api_key,
                timeout=30,
                caching=CACHE_ENABLED,
            )
        except Exception as exc:
            if _is_retriable(exc):
                logger.warning("Key %s failed (%s), trying next key", key_id, exc)
                fallback_attempts += 1
                continue
            logger.error("Non-retriable error on key %s: %s", key_id, exc)
            fallback_attempts += 1
            continue

        # ── Detect cache hit ───────────────────────────────────────────────
        # LiteLLM sets _hidden_params["cache_hit"] = True on cached responses
        is_cache_hit = False
        try:
            is_cache_hit = bool(
                getattr(response, "_hidden_params", {}).get("cache_hit", False)
            )
        except Exception:
            pass

        reply = response.choices[0].message.content or ""
        model_name = getattr(response, "model", None) or LLM_MODEL

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        if is_cache_hit:
            logger.info("Cache hit")
            return CacheHit(
                reply=reply,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return {
            "reply": reply,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "key_id": key_id,
            "fallback_attempts": fallback_attempts,
        }

    logger.error("All %d keys failed", len(KEY_POOL))
    return None
