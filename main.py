"""
AI Agent — entry point.

Minimal (local UI only):
  ANTHROPIC_API_KEY=... python main.py   →  http://localhost:8000

Full stack (Telegram + Vapi + ElevenLabs):
  Fill in .env, then: python main.py
"""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from modules import config
from modules.context_manager import ContextManager
from modules.ui_handlers import register_ui, push_ui_event

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

# Telegram is optional — only initialised if TELEGRAM_BOT_TOKEN is set
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
    logger.info("Telegram bot initialised")
else:
    logger.info("TELEGRAM_BOT_TOKEN not set — Telegram disabled, using local UI only")

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    context_manager.load()
    logger.info("Context loaded. UI ready at http://localhost:%d", config.PORT)
    yield


fastapi_app = FastAPI(title="AI Agent", lifespan=lifespan)

# Serve synthesised audio files at /audio/<filename>.mp3
config.AUDIO_CACHE_DIR.mkdir(exist_ok=True)
fastapi_app.mount("/audio", StaticFiles(directory=str(config.AUDIO_CACHE_DIR)), name="audio")

# Local chat UI (replaces Telegram as the dev/test interface)
register_ui(fastapi_app, context_manager)


@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------
# Vapi routes
# ------------------------------------------------------------------
@fastapi_app.post("/vapi-llm")
@fastapi_app.post("/vapi-llm/chat/completions")
async def vapi_llm(request: Request):
    """Vapi Custom LLM endpoint — handles live call conversations via Claude."""
    from modules.vapi_handler import handle_vapi_llm
    return await handle_vapi_llm(request, context_manager)


@fastapi_app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """
    Optional Vapi post-call webhook.
    UI calls are handled via polling (no public endpoint required).
    This only fires for Telegram-initiated calls when running with a public URL.
    """
    payload = await request.json()
    msg = payload.get("message", {})
    event_type = msg.get("type", "unknown")
    logger.info("Vapi webhook: %s", event_type)

    if event_type == "end-of-call-report" and telegram_app is not None:
        asyncio.create_task(_handle_telegram_call_ended(msg))

    return Response(status_code=200)


async def _handle_telegram_call_ended(msg: dict) -> None:
    """Delivers Telegram-initiated call results back to the Telegram chat."""
    import modules.vapi_client as vapi_client

    call_id = msg.get("call", {}).get("id", "")
    transcript = msg.get("transcript", "")
    ended_reason = msg.get("endedReason", "")
    call_info = vapi_client.pop_call(call_id)

    if not call_info or not transcript or telegram_app is None:
        return

    # Skip if this call was initiated from the UI (handled by polling instead)
    if call_info.get("chat_id") == config.UI_CHAT_ID:
        return

    chat_id = call_info["chat_id"]
    purpose = call_info["purpose"]

    import anthropic as _anthropic
    client = _anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    try:
        resp = await client.messages.create(
            model=config.CLAUDE_CALL_MODEL,
            max_tokens=300,
            system=(
                "You are extracting the result of a phone call made by Manie's AI agent. "
                "Read the transcript and answer the original mission in 1-2 sentences. "
                "Be direct: start with Sim/Não if the answer is yes/no, then a brief explanation. "
                "Reply in Portuguese."
            ),
            messages=[{
                "role": "user",
                "content": f"Mission: {purpose}\n\nTranscript:\n{transcript}"
            }],
        )
        answer = resp.content[0].text if resp.content else "Sem resposta."
    except Exception as exc:
        logger.error("Error extracting call result: %s", exc)
        answer = "Chamada terminada, mas não foi possível extrair a resposta."

    await telegram_app.bot.send_message(
        chat_id=chat_id,
        text=f"📞 *Resultado da chamada:*\n\n{answer}\n\n_Motivo do fim: {ended_reason}_",
        parse_mode="Markdown",
    )


# ------------------------------------------------------------------
# Telegram webhook (production mode — when WEBHOOK_URL is set)
# ------------------------------------------------------------------
@fastapi_app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if telegram_app is None:
        return Response(status_code=404)
    from telegram import Update
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
async def _run_telegram_polling():
    if telegram_app is None:
        return
    logger.info("Starting Telegram polling...")
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
            host="0.0.0.0",
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
