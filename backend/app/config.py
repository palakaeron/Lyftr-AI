"""
Configuration loaded from environment variables.
"""
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── LLM keys ──────────────────────────────────────────────────────────────────
_raw_keys = os.getenv("LLM_API_KEYS", "")
LLM_API_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]

if not LLM_API_KEYS:
    print(
        "WARNING: LLM_API_KEYS is not set or empty. "
        "The /chat endpoint will return 503 until keys are configured.",
        file=sys.stderr,
    )

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:////data/app.db")

# Strip the sqlite:/// or sqlite://// prefix to get the file path
if DATABASE_URL.startswith("sqlite:////"):
    DB_PATH: str = DATABASE_URL[len("sqlite:///"):]
elif DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = DATABASE_URL[len("sqlite:///"):]
else:
    DB_PATH = DATABASE_URL

# ── CORS ───────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_TYPE: str = os.getenv("CACHE_TYPE", "disk")  # "disk" or "in-memory"
CACHE_DIR: str = os.getenv("CACHE_DIR", "/data/litellm_cache")

# ── LLM model ─────────────────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-120b")

# ── Failure simulation ────────────────────────────────────────────────────────
ENABLE_FAILURE_SIMULATION: bool = (
    os.getenv("ENABLE_FAILURE_SIMULATION", "false").lower() == "true"
)
