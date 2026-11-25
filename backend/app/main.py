"""
Lyftr AI Assignment — FastAPI backend
"""
import os
import time
import uuid
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from .database import init_db, get_db, save_message, get_messages, list_sessions
from .llm import chat_completion, CacheHit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Database initialised")
    yield


app = FastAPI(title="Lyftr AI Chatbot", lifespan=lifespan)

allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 128:
            raise ValueError("session_id must be 1–128 characters")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        if len(v) > 4000:
            raise ValueError("message must not exceed 4000 characters")
        return v


class MessageOut(BaseModel):
    role: str
    content: str
    metadata: Optional[dict] = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageOut]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_simulate_header(header_value: Optional[str]) -> dict:
    """Parse X-Simulate-Failure header into a dict of {key_id: status_code}."""
    if not header_value:
        return {}
    result = {}
    for part in header_value.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        key, code_str = part.split("=", 1)
        try:
            result[key.strip()] = int(code_str.strip())
        except ValueError:
            pass
    return result


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from .llm import KEY_POOL
    from .database import check_db

    db_ok = check_db()
    if not db_ok:
        raise HTTPException(status_code=503, detail="database unavailable")

    from .config import CACHE_ENABLED
    return {
        "status": "ok",
        "keys_configured": len(KEY_POOL),
        "cache_enabled": CACHE_ENABLED,
    }


@app.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
):
    simulation_map: dict = {}
    enable_sim = os.getenv("ENABLE_FAILURE_SIMULATION", "false").lower() == "true"
    if enable_sim and x_simulate_failure:
        simulation_map = _parse_simulate_header(x_simulate_failure)

    # Retrieve existing history (None = new session, treat as empty)
    history = get_messages(req.session_id) or []

    # Build messages list for LLM
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.message})

    start = time.perf_counter()
    result = await chat_completion(messages, simulation_map)
    latency_ms = int((time.perf_counter() - start) * 1000)

    if result is None:
        raise HTTPException(status_code=503, detail="all providers unavailable")

    is_cache: bool = isinstance(result, CacheHit)

    if is_cache:
        resolution_source = "LLM_CACHE"
        reply = result.reply
        model = result.model
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        key_id = None
        fallback_attempts = 0
    else:
        resolution_source = "LIVE_INFERENCE"
        reply = result["reply"]
        model = result["model"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        key_id = result["key_id"]
        fallback_attempts = result["fallback_attempts"]

    metadata = {
        "resolution_source": resolution_source,
        "model": model,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "key_id": key_id,
        "fallback_attempts": fallback_attempts,
    }

    # Persist user message and assistant reply
    save_message(req.session_id, "user", req.message, None)
    save_message(req.session_id, "assistant", reply, metadata)

    return {
        "session_id": req.session_id,
        "reply": reply,
        "metadata": metadata,
    }


@app.get("/sessions")
async def get_all_sessions():
    """
    Returns all sessions ordered by most recent activity.
    Each item includes session_id, a short preview from the first user message,
    and the timestamp of the last message in that session.
    """
    sessions = list_sessions()
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    rows = get_messages(session_id)
    if rows is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": r["role"],
                "content": r["content"],
                "metadata": r.get("metadata"),
            }
            for r in rows
        ],
    }


# ── Global error handler ───────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
