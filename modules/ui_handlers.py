"""
Local development chat UI.
Replaces Telegram as the control interface — same full stack (Claude + Vapi + ElevenLabs),
controlled from a browser at http://localhost:8000.

Routes
------
GET  /                → chat UI (static/index.html)
GET  /api/status      → feature flags + active model
GET  /api/events      → SSE stream (call results, system messages)
POST /api/chat        → send a message; /call commands trigger the full call flow
DELETE /api/history   → clear conversation history
POST /api/refresh     → reload context files from disk
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from modules import claude_client, config, vapi_client
from modules.context_manager import ContextManager

logger = logging.getLogger(__name__)

_PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")
_STATIC_HTML = Path(__file__).parent.parent / "static" / "index.html"

# Single-user SSE queue — localhost only
_events: asyncio.Queue = asyncio.Queue(maxsize=100)


def push_ui_event(event: dict) -> None:
    """Push an event to the browser SSE stream. Safe to call from any async context."""
    try:
        _events.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning("UI event queue full — dropping event: %s", event.get("type"))


def register_ui(app: FastAPI, context_manager: ContextManager) -> None:
    """Attach all local-UI routes to the FastAPI app."""

    # ── Serve the HTML shell ──────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        if _STATIC_HTML.exists():
            return HTMLResponse(_STATIC_HTML.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=404)

    # ── Status ───────────────────────────────────────────────────────

    @app.get("/api/status")
    async def api_status():
        return {
            "model": config.CLAUDE_CHAT_MODEL,
            "elevenlabs_enabled": config.ELEVENLABS_ENABLED,
            "vapi_enabled": config.VAPI_ENABLED,
        }

    # ── Server-Sent Events ────────────────────────────────────────────

    @app.get("/api/events")
    async def api_events(request: Request):
        """
        Browser connects here once and keeps the connection open.
        Call results and other async events are pushed here in real time.
        """
        async def stream() -> AsyncGenerator[str, None]:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(_events.get(), timeout=25)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep the connection alive

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Chat / Call ───────────────────────────────────────────────────

    @app.post("/api/chat")
    async def api_chat(request: Request):
        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        # Route /call commands to the full call flow
        if message.lower().startswith("/call"):
            return await _handle_call_command(message[5:].strip(), context_manager)

        # Regular Q&A chat with Claude
        system_prompt = context_manager.get_system_prompt()
        knowledge_string = context_manager.get_knowledge_string()

        result = await asyncio.to_thread(
            claude_client.chat,
            chat_id=config.UI_CHAT_ID,
            user_message=message,
            system_prompt=system_prompt,
            knowledge_string=knowledge_string,
        )

        if not result["success"]:
            return JSONResponse(
                {"error": result.get("error", "Unknown error")}, status_code=500
            )

        return JSONResponse(
            {
                "text": result.get("text", ""),
                "tool_call": result.get("tool_call"),
            }
        )

    # ── Utility ───────────────────────────────────────────────────────

    @app.delete("/api/history")
    async def api_clear_history():
        claude_client.clear_history(config.UI_CHAT_ID)
        return JSONResponse({"ok": True})

    @app.post("/api/refresh")
    async def api_refresh():
        context_manager.refresh()
        return JSONResponse({"ok": True})


# ── Call flow (mirrors telegram_handlers._handle_call_command) ────────────────

async def _handle_call_command(args: str, context_manager: ContextManager) -> JSONResponse:
    """
    Parse a /call command, generate a script via Claude, synthesise audio,
    initiate the Vapi call, and return immediately.
    The end-of-call result arrives later via SSE (pushed by _handle_call_ended in main.py).
    """
    if not args:
        return JSONResponse(
            {"error": "Usage: /call <phone> [reason]"}, status_code=400
        )

    phone_match = _PHONE_RE.search(args)
    if not phone_match:
        return JSONResponse(
            {"error": "No valid phone number found. Example: /call +351912345678 reason"},
            status_code=400,
        )

    phone = re.sub(r"[\s\-()]", "", phone_match.group(1))
    purpose = args.replace(phone_match.group(1), "").strip() or "assunto não especificado"

    # Ask Claude (haiku) to generate the call script — same prompt as Telegram handler
    system_prompt = context_manager.get_system_prompt()
    knowledge_string = context_manager.get_knowledge_string()
    call_intent = (
        f"Faz uma chamada para {phone}.\n"
        f"Missão: {purpose}\n\n"
        "Gera um first_message ultra-curto e direto para quando a chamada atender. "
        "O call_script deve conter APENAS o necessário para obter a resposta à missão. "
        "Inclui no call_context todos os dados do cliente mencionados (NIF, morada, email, telemóvel) "
        "para que o agente possa responder a perguntas de verificação de identidade."
    )

    result = await asyncio.to_thread(
        claude_client.chat,
        chat_id=config.UI_CHAT_ID,
        user_message=call_intent,
        system_prompt=system_prompt,
        knowledge_string=knowledge_string,
        model=config.CLAUDE_CALL_MODEL,
    )

    if not result["success"]:
        return JSONResponse(
            {"error": f"Script generation failed: {result.get('error')}"}, status_code=500
        )

    tool_call = result.get("tool_call")
    if not tool_call or tool_call["name"] != "initiate_call":
        # Claude responded with text instead of a tool call
        return JSONResponse(
            {
                "type": "chat",
                "text": result.get("text") or "Claude did not generate a call script.",
            }
        )

    tool_input = tool_call["input"]
    script: str = tool_input.get("call_script", "")
    call_phone: str = tool_input.get("phone_number", phone)
    first_message: str = tool_input.get("first_message", "")

    # Synthesise audio + initiate call in parallel (same pattern as Telegram handler)
    from modules import elevenlabs_client

    audio_task = asyncio.create_task(elevenlabs_client.synthesize_async(script))
    call_task = asyncio.create_task(
        vapi_client.initiate_call_async(
            to_number=call_phone,
            call_context=script,
            first_message=first_message,
        )
    )
    audio_result, call_result = await asyncio.gather(audio_task, call_task)

    call_id = call_result.get("call_id") if call_result["success"] else None

    # Keep Claude's conversation history consistent by submitting the tool result
    status_text = (
        f"Call to {call_phone}: {'initiated' if call_result['success'] else 'failed'}. "
        f"Script audio: {'ready' if audio_result['success'] else 'failed'}."
    )
    await asyncio.to_thread(
        claude_client.submit_tool_result,
        chat_id=config.UI_CHAT_ID,
        tool_use_id=tool_call["id"],
        result_text=status_text,
    )

    # Start background polling — no public webhook needed
    if call_id:
        asyncio.create_task(
            _poll_and_deliver_result(call_id, call_phone, purpose)
        )

    return JSONResponse(
        {
            "type": "call_initiated",
            "phone": call_phone,
            "purpose": purpose,
            "script": script,
            "first_message": first_message,
            "call_id": call_id,
            "call_error": call_result.get("error") if not call_result["success"] else None,
            "audio_url": audio_result.get("audio_url") if audio_result["success"] else None,
            "audio_error": audio_result.get("error") if not audio_result["success"] else None,
        }
    )


async def _poll_and_deliver_result(call_id: str, phone: str, purpose: str) -> None:
    """
    Background task: polls Vapi every 5s until the call ends, then extracts
    a concise answer via Claude and pushes everything to the browser via SSE.
    No public endpoint required — pure outbound polling.
    """
    logger.info("Polling Vapi for call %s…", call_id)
    call = await vapi_client.poll_call_until_ended(call_id)

    if call is None:
        push_ui_event({
            "type": "call_ended",
            "call_id": call_id,
            "answer": "Chamada atingiu o tempo limite sem resposta.",
            "ended_reason": "timeout",
            "transcript": "",
            "recording_url": "",
        })
        return

    transcript: str = call.get("transcript", "")
    ended_reason: str = call.get("endedReason", "")

    # Vapi may nest the recording URL inside artifact or at the top level
    artifact = call.get("artifact") or {}
    recording_url: str = (
        artifact.get("recordingUrl")
        or artifact.get("videoRecordingUrl")
        or call.get("recordingUrl")
        or ""
    )

    # Ask Claude to extract a concise answer from the transcript
    answer = await _extract_call_answer(purpose, transcript)

    push_ui_event({
        "type": "call_ended",
        "call_id": call_id,
        "phone": phone,
        "answer": answer,
        "ended_reason": ended_reason,
        "transcript": transcript,
        "recording_url": recording_url,
    })
    logger.info("Call %s ended — result pushed to UI", call_id)


async def _extract_call_answer(purpose: str, transcript: str) -> str:
    """Use Claude to extract a 1-2 sentence answer from the call transcript."""
    if not transcript:
        return "Sem transcrição disponível."

    import anthropic
    client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
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
                "content": f"Mission: {purpose}\n\nTranscript:\n{transcript}",
            }],
        )
        return resp.content[0].text if resp.content else "Sem resposta."
    except Exception as exc:
        logger.error("Error extracting call result: %s", exc)
        return "Chamada terminada, mas não foi possível extrair a resposta."
