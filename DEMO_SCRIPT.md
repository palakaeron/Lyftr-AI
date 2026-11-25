# Demo Recording Script — Lyftr AI Chatbot
# 2–3 minute walkthrough

## Setup before recording

Ensure the backend is running with real API keys:

```bash
# Edit .env — set your real OpenAI key(s)
# LLM_API_KEYS=sk-proj-...actual_key_1...,sk-proj-...actual_key_2...
# ENABLE_FAILURE_SIMULATION=true

make up
# Wait ~20 seconds for health checks to pass
```

Open two terminal windows side-by-side.

---

## STEP 1 — Open the deployed chatbot (0:00 – 0:20)

Navigate to: http://localhost:3000

Point out the interface:
- System prompt input (optional)
- Chat input field
- Send button

---

## STEP 2 — Send a message, show LIVE_INFERENCE (0:20 – 0:50)

In the UI, type:

> "What is the capital of France?"

Send it. After reply, point out the metadata badge below the response:
- Resolution source: **LIVE_INFERENCE**
- Model, latency, input/output tokens
- key_id: key_1 (or key_2)

Verify via terminal:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-s1","message":"What is the capital of France?"}' | python -m json.tool
```

---

## STEP 3 — Show LLM_CACHE (0:50 – 1:20)

Refresh the browser (new session ID is generated automatically).

Type the **same first message** in the new session:

> "What is the capital of France?"

The metadata badge should now show:
- Resolution source: **LLM_CACHE**
- key_id: null

This confirms LiteLLM's cache matched on the identical message list.

Verify via terminal (uses a fresh session ID):

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-s2","message":"What is the capital of France?"}' \
  | python -m json.tool
```

---

## STEP 4 — Multi-turn conversation memory (1:20 – 1:50)

In the same browser session (do NOT refresh), send:

> "What language do they speak there?"

The AI responds in context ("They speak French in Paris/France"), demonstrating that the backend maintains session history. The client only sent the new message.

Then send:

> "Name three famous landmarks."

The AI responds with French landmarks (Eiffel Tower, Louvre, etc.), still in context.

---

## STEP 5 — Trigger key_1=429 failover (1:50 – 2:15)

Run in terminal:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"demo-failover","message":"Hello, test failover"}' \
  | python -m json.tool
```

Expected response:
- status 200
- metadata.key_id = "key_2"
- metadata.fallback_attempts = 1

This shows key_1 was simulated to fail with 429 and the backend automatically fell over to key_2.

---

## STEP 6 — Trigger *=429 (all keys fail → 503) (2:15 – 2:35)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"demo-allfail","message":"This should fail"}' \
  -w "\nHTTP Status: %{http_code}\n"
```

Expected: HTTP 503 `{"detail":"all providers unavailable"}`

Verify that session was not stored:

```bash
curl -s http://localhost:8000/sessions/demo-allfail/messages
# Expected: 404 {"detail":"session not found"}
```

---

## STEP 7 — Show the three API endpoints (2:35 – 3:00)

**Health endpoint:**

```bash
curl -s http://localhost:8000/health | python -m json.tool
# {"status":"ok","keys_configured":2,"cache_enabled":true}
```

**Chat history endpoint:**

```bash
curl -s http://localhost:8000/sessions/demo-s1/messages | python -m json.tool
# Shows stored user and assistant messages with metadata
```

**FastAPI docs (interactive):**

Navigate to: http://localhost:8000/docs

Point out the three endpoints: GET /health, POST /chat, GET /sessions/{session_id}/messages.

---

## Wrap-up (3:00)

Summary of what was demonstrated:
1. ✅ LIVE_INFERENCE on first call
2. ✅ LLM_CACHE on identical repeat
3. ✅ Multi-turn memory (server-side, client sends only new message)
4. ✅ Automatic key failover (key_1=429 → key_2)
5. ✅ All-keys-failed returns 503, nothing stored in DB
6. ✅ All three API endpoints working
