"""
Environment variable loading and fail-fast validation.
All other modules import constants from here.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

REQUIRED_VARS = [
    "ANTHROPIC_API_KEY",
]


def validate() -> None:
    missing = [k for k in REQUIRED_VARS if not os.getenv(k)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        raise SystemExit(1)


# Telegram
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Anthropic
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_CHAT_MODEL: str = os.getenv("CLAUDE_CHAT_MODEL", "claude-opus-4-6")
CLAUDE_CALL_MODEL: str = os.getenv("CLAUDE_CALL_MODEL", "claude-haiku-4-5-20251001")

# Vapi
VAPI_API_KEY: str = os.getenv("VAPI_API_KEY", "")
VAPI_ASSISTANT_ID: str = os.getenv("VAPI_ASSISTANT_ID", "")
VAPI_PHONE_NUMBER_ID: str = os.getenv("VAPI_PHONE_NUMBER_ID", "")

# ElevenLabs
ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID: str = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# Live API
CRM_API_BASE_URL: str = os.getenv("CRM_API_BASE_URL", "")
CRM_API_TOKEN: str = os.getenv("CRM_API_TOKEN", "")

# App
APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8000")
PORT: int = int(os.getenv("PORT", "8000"))

# Feature flags — derived from which env vars are set
TELEGRAM_ENABLED: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
ELEVENLABS_ENABLED: bool = bool(os.getenv("ELEVENLABS_API_KEY") and os.getenv("ELEVENLABS_VOICE_ID"))
VAPI_ENABLED: bool = bool(
    os.getenv("VAPI_API_KEY")
    and os.getenv("VAPI_PHONE_NUMBER_ID")
)

# Paths
BASE_DIR = Path(__file__).parent.parent
CONTEXT_DIR = BASE_DIR / "context"
AUDIO_CACHE_DIR = BASE_DIR / "audio_cache"

# Reserved chat session ID for the local UI (distinguishes from Telegram chat IDs)
UI_CHAT_ID: int = 0
