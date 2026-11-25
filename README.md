# Lyftr AI — LLM Chatbot

A full-stack LLM chatbot built with **React, FastAPI, LiteLLM, and SQLite**. Supports multi-turn conversations, automatic API-key failover, response caching, and per-request token tracking.

---

## 1. Live URLs

| Service | URL |
|---|---|
| Frontend | `https://lyftr-ai-pied-two.vercel.app/` |
| Backend | `https://lyftr-ai-33pr.onrender.com` |
| API docs | `https://lyftr-ai-33pr.onrender.com/docs` |
| Repository | `https://github.com/palakaeron/Lyftr-AI` |

---

## 2. Run locally: with Docker and without

### Prerequisites

- Docker Desktop (for the Docker path)
- Python 3.11+ and Node.js 20+ (for the non-Docker path)
- At least two API keys for your LLM provider

### With Docker

```bash
cp .env.example .env
# Fill in LLM_API_KEYS with real provider keys

make up
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

```bash
make down   # stop and remove containers (data volume is kept)
```

The named Docker volume `chatdata` is mounted at `/data` inside both services. It holds the SQLite database (`/data/app.db`) and the LiteLLM disk cache (`/data/litellm_cache`). The volume survives `make down`.

### Without Docker

**Backend**

```bash
cp .env.example .env
# Set DATABASE_URL=sqlite:///./data/app.db  (and your LLM_API_KEYS)

cd backend
pip install -r requirements.txt
mkdir -p data
uvicorn app.main:app --reload --port 8000
```

**Frontend** (second terminal)

```bash
cd frontend
npm install --legacy-peer-deps
# Create frontend/.env.local:
#   REACT_APP_API_URL=http://localhost:8000
npm start
```

### Environment variables

| Variable | Purpose |
|---|---|
| `LLM_API_KEYS` | Comma-separated provider keys — at least 2. IDs are `key_1`, `key_2`, … |
| `LLM_MODEL` | LiteLLM model string, e.g. `groq/llama3-8b-8192` |
| `DATABASE_URL` | SQLite URL, e.g. `sqlite:////data/app.db` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins for the frontend |
| `ENABLE_FAILURE_SIMULATION` | `true` to allow `X-Simulate-Failure` on the deployed backend |
| `CACHE_ENABLED` | `true` / `false` |
| `CACHE_TYPE` | `disk` (default) or `in-memory` |
| `CACHE_DIR` | Directory for the disk cache, default `/data/litellm_cache` |

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Key fallback

Each key in `LLM_API_KEYS` receives a stable identifier (`key_1`, `key_2`, …). On every `/chat` request the backend iterates through the pool in order. If a key returns HTTP 429, 500, 502, 503, a connection error, or a timeout, `fallback_attempts` is incremented and the next key is tried. The first key that succeeds is recorded in the response as `key_id`.

If every key fails the endpoint returns `HTTP 503 {"detail": "all providers unavailable"}` and nothing is written to the database.

The `X-Simulate-Failure` header (enabled only when `ENABLE_FAILURE_SIMULATION=true`) injects per-request artificial failures for specific key IDs or for all keys (`*=429`). Simulated failures go through the same fallback code path, so the successful key still calls the real LLM provider.

### Caching

LiteLLM's built-in `Cache` is configured with a disk backend (`/data/litellm_cache`) so entries survive backend restarts. The cache key is a SHA-256 hash of the **complete message list** sent to the model — model name, system prompt, all prior turns, and the new user message.

Because each session starts with an empty history, an identical first message in two different sessions produces the same message list and therefore the same cache key. The second session gets `resolution_source: "LLM_CACHE"` with `key_id: null` and the original token counts. Within a session, repeating the same question always misses the cache because the growing conversation history changes the input.

LiteLLM sets `response._hidden_params["cache_hit"] = True` on cache hits; the backend detects this flag and returns the `CacheHit` dataclass.

### SQLite session storage

Each chat message (user and assistant) is stored in a `messages` table keyed by `session_id`. Before every LLM call the backend loads the full history for that session and includes it in the request, so multi-turn context is maintained entirely server-side — the client only sends the new message.

The user message and assistant reply are written **only after a successful LLM call**. Failed requests (all keys unavailable) are never written, so incomplete exchanges do not appear in the history.

A WAL-mode SQLite connection is used to allow concurrent reads without blocking writes. In Docker the database lives at `/data/app.db` on the named `chatdata` volume.

---

## 4. Verification: how to see failover, cache hits and per-message token usage

All commands target `http://localhost:8000`. Replace with the deployed backend URL for live verification.

### Cache hit

Send the same first message from two different sessions:

```bash
# First session — live call
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"What is 2+2?"}' \
  | jq .metadata.resolution_source
# → "LIVE_INFERENCE"

# Second session — same message list → cache hit
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s2","message":"What is 2+2?"}' \
  | jq .metadata.resolution_source
# → "LLM_CACHE"
```

### Key failover (`key_1=429`)

Requires `ENABLE_FAILURE_SIMULATION=true`.

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s3","message":"Test failover"}' \
  | jq '.metadata | {key_id, fallback_attempts}'
# → { "key_id": "key_2", "fallback_attempts": 1 }
```

### All-provider failure (503)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s4","message":"Test total failure"}'
# → HTTP 503  {"detail":"all providers unavailable"}
```

Confirm nothing was stored:

```bash
curl -s http://localhost:8000/sessions/s4/messages
# → HTTP 404  {"detail":"session not found"}
```

### Per-message token usage

```bash
curl -s http://localhost:8000/sessions/s1/messages \
  | jq '.messages[] | {role, input_tokens: .metadata.input_tokens, output_tokens: .metadata.output_tokens}'
```

### Health check

```bash
curl -s http://localhost:8000/health
# → {"status":"ok","keys_configured":2,"cache_enabled":true}
```

### Backend tests

```bash
cd backend
pip install -r requirements.txt -r requirements-test.txt
pytest tests/ -v
# 22 passed
```

---

## 5. Setup Used

- **Editor:** VS Code with GitHub Copilot
- **AI assistance:** Kiro AI — used for scaffolding, architecture review, and implementation iterations
- **Runtime:** Python 3.11, Node.js 20, Docker Desktop
- **Key libraries:** FastAPI 0.111, LiteLLM 1.110, Pydantic v2, React 18, pytest 8

---

## 6. Bonus features

- **Disk-persisted LiteLLM cache** — stored on the Docker volume, so repeated identical queries stay fast across backend restarts.
- **SQLite WAL mode** — allows concurrent reads without blocking writes.
- **Failure simulation at any granularity** — `X-Simulate-Failure` accepts specific key IDs or the wildcard `*`, making it easy to demonstrate partial and total outage scenarios without changing any configuration.
- **Non-root Docker container** — the backend runs as `appuser`, not root.
- **New Chat control** — sidebar button starts a fresh session without a page reload.
- **Session history browser** — the sidebar lists all past conversations retrieved from `GET /sessions`, ordered by most recent activity. Each entry shows the first user message as a preview. Clicking any entry restores the full conversation via `GET /sessions/{id}/messages`. New sessions appear in the list the moment their first reply arrives.
