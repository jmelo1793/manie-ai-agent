"""
AI Agent — entry point.

  python main.py  →  http://localhost:8000

The server binds ONLY to 127.0.0.1 — it is not reachable from any other
machine, network interface, or the internet.

All Vapi results are delivered via outbound polling (this app polls Vapi's
API, not the other way around). No public endpoint, no ngrok needed.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from modules import config
from modules.context_manager import ContextManager
from modules.ui_handlers import register_ui

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(config.BASE_DIR / "ai_agent.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Fail fast on missing required env vars (only ANTHROPIC_API_KEY)
# ------------------------------------------------------------------
config.validate()

# ------------------------------------------------------------------
# Shared singletons
# ------------------------------------------------------------------
context_manager = ContextManager()

# Telegram is optional — only initialised if TELEGRAM_BOT_TOKEN is set.
# When enabled, uses long-polling (outbound from this app to Telegram servers).
# No inbound webhook is registered.
telegram_app = None
if config.TELEGRAM_ENABLED:
    from telegram.ext import ApplicationBuilder
    from modules.telegram_handlers import register_handlers

    telegram_app = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )
    register_handlers(telegram_app, context_manager)
    logger.info("Telegram bot initialised (polling mode — no public endpoint)")
else:
    logger.info("TELEGRAM_BOT_TOKEN not set — Telegram disabled")

# ------------------------------------------------------------------
# FastAPI — binds to 127.0.0.1 ONLY
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    context_manager.load()
    logger.info("Context loaded. UI ready at http://localhost:%d", config.PORT)
    yield


fastapi_app = FastAPI(title="AI Agent", lifespan=lifespan)

# Serve synthesised audio files at /audio/<filename>.mp3 (localhost only)
config.AUDIO_CACHE_DIR.mkdir(exist_ok=True)
fastapi_app.mount("/audio", StaticFiles(directory=str(config.AUDIO_CACHE_DIR)), name="audio")

# Local chat UI
register_ui(fastapi_app, context_manager)


@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}


# NOTE: No inbound webhook routes.
# - Vapi results → delivered by polling (ui_handlers._poll_and_deliver_result)
# - Telegram updates → delivered by long-polling (_run_telegram_polling below)
# Nothing from the outside world can POST to this server.


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def _run_telegram_polling():
    """Outbound polling — this app connects to Telegram, not the other way around."""
    if telegram_app is None:
        return
    logger.info("Starting Telegram polling (outbound, no public URL needed)…")
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot running.")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()


async def _run_all():
    context_manager.load()
    server = uvicorn.Server(
        uvicorn.Config(
            app=fastapi_app,
            host="127.0.0.1",   # localhost ONLY — not reachable from outside this machine
            port=config.PORT,
            log_level="warning",
        )
    )
    tasks = [server.serve()]
    if telegram_app is not None:
        tasks.append(_run_telegram_polling())
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(_run_all())
