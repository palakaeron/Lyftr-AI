"""
Controlled end-to-end cache behaviour test.

Verifies:
  1. Session A turn 1  → LIVE_INFERENCE   (1 provider call)
  2. Session A turn 2  → LIVE_INFERENCE   (2 provider calls: history grew → different cache key)
  3. Session B turn 1  → LLM_CACHE        (still 2 provider calls: LLM input identical to A-turn-1)

Run: python test_cross_session.py
"""
import asyncio
import os
import sys
import logging

os.environ.update({
    'CACHE_ENABLED': 'true',
    'CACHE_TYPE': 'in-memory',
    'LLM_API_KEYS': 'sk-key-one,sk-key-two',
    'LLM_MODEL': 'gpt-3.5-turbo',
    'DATABASE_URL': 'sqlite:///./data/diag.db',
    'ENABLE_FAILURE_SIMULATION': 'false',
})
sys.path.insert(0, '.')

for name in ['LiteLLM', 'LiteLLM Proxy', 'app.llm', 'app.main', 'app.database',
             'httpx', 'httpcore', 'asyncio', 'starlette', 'uvicorn']:
    logging.getLogger(name).setLevel(logging.ERROR)

import sqlite3
from app.database import init_db, DB_PATH
conn = sqlite3.connect(DB_PATH)
conn.execute('DROP TABLE IF EXISTS messages')
conn.commit()
conn.close()
init_db()

import litellm
from litellm import Cache
from litellm.utils import ModelResponse
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app

PATCH = 'litellm.llms.openai.openai.OpenAIChatCompletion.acompletion'


def _response():
    return ModelResponse(
        id='x',
        choices=[{'message': {'role': 'assistant', 'content': '4'}, 'finish_reason': 'stop', 'index': 0}],
        model='gpt-3.5-turbo',
        usage={'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12},
    )


async def run() -> bool:
    litellm.cache = Cache(type='local')
    litellm.enable_cache()

    provider_calls = {'n': 0}
    failures = []

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:

        # ── Test 1: Session A, turn 1 ─────────────────────────────────────────
        with patch(PATCH, new=AsyncMock(
            side_effect=lambda *a, **kw: (
                provider_calls.__setitem__('n', provider_calls['n'] + 1), _response()
            )[1]
        )):
            r1 = await client.post('/chat', json={'session_id': 'sess-A', 'message': 'What is 2+2?'})
        # Yield so litellm's background cache-write task can complete
        for _ in range(60):
            await asyncio.sleep(0)

        m1 = r1.json()['metadata']
        ok1 = m1['resolution_source'] == 'LIVE_INFERENCE' and provider_calls['n'] == 1
        print(f"[{'PASS' if ok1 else 'FAIL'}] Test 1 — Session A, first question")
        print(f"       messages sent to LLM : [user: 'What is 2+2?']")
        print(f"       resolution_source    : {m1['resolution_source']}  (expected LIVE_INFERENCE)")
        print(f"       provider calls       : {provider_calls['n']}  (expected 1)")
        if not ok1:
            failures.append('Test 1')

        # ── Test 2: Session A, same question again ─────────────────────────────
        # LLM input is now [user: Q, assistant: A, user: Q] — different cache key
        with patch(PATCH, new=AsyncMock(
            side_effect=lambda *a, **kw: (
                provider_calls.__setitem__('n', provider_calls['n'] + 1), _response()
            )[1]
        )):
            r2 = await client.post('/chat', json={'session_id': 'sess-A', 'message': 'What is 2+2?'})
        for _ in range(60):
            await asyncio.sleep(0)

        m2 = r2.json()['metadata']
        ok2 = m2['resolution_source'] == 'LIVE_INFERENCE' and provider_calls['n'] == 2
        print(f"\n[{'PASS' if ok2 else 'FAIL'}] Test 2 — Session A, same question (history now contains turn 1)")
        print(f"       messages sent to LLM : [user: Q, assistant: A, user: Q]")
        print(f"       resolution_source    : {m2['resolution_source']}  (expected LIVE_INFERENCE — history changed cache key)")
        print(f"       provider calls       : {provider_calls['n']}  (expected 2)")
        if not ok2:
            failures.append('Test 2')

        # ── Test 3: Session B, same first question ─────────────────────────────
        # Fresh session → LLM input is [user: Q], identical to A-turn-1 → cache HIT
        with patch(PATCH, new=AsyncMock(
            side_effect=lambda *a, **kw: (
                provider_calls.__setitem__('n', provider_calls['n'] + 1), _response()
            )[1]
        )):
            r3 = await client.post('/chat', json={'session_id': 'sess-B', 'message': 'What is 2+2?'})
        for _ in range(60):
            await asyncio.sleep(0)

        m3 = r3.json()['metadata']
        ok3 = (
            m3['resolution_source'] == 'LLM_CACHE'
            and provider_calls['n'] == 2
            and m3['key_id'] is None
        )
        print(f"\n[{'PASS' if ok3 else 'FAIL'}] Test 3 — Session B, identical first question (new session)")
        print(f"       messages sent to LLM : [user: 'What is 2+2?']  (identical to Test 1)")
        print(f"       resolution_source    : {m3['resolution_source']}  (expected LLM_CACHE)")
        print(f"       key_id               : {m3['key_id']}  (expected null — cache hits have no key_id)")
        print(f"       provider calls       : {provider_calls['n']}  (expected 2 — no extra provider call)")
        if not ok3:
            failures.append('Test 3')

    print()
    if failures:
        print(f"OVERALL: FAIL — {failures}")
        return False
    else:
        print("OVERALL: PASS — all 3 tests passed")
        return True


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    try:
        ok = loop.run_until_complete(run())
    finally:
        # Cancel any stray background tasks (e.g. litellm LoggingWorker) so the
        # process exits cleanly without a multi-second hang.
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    sys.exit(0 if ok else 1)
