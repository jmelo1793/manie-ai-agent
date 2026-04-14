"""
Retell Custom LLM WebSocket handler.
Retell connects here during a call, sends transcriptions, receives Claude responses.

Protocol:
  Retell → server: {"interaction_type": "call_started" | "response_required" | "reminder_required", ...}
  server → Retell: {"response_type": "response", "content": "...", "content_complete": true}
"""
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from modules import claude_client
from modules.context_manager import ContextManager

logger = logging.getLogger(__name__)


async def handle_retell_websocket(websocket: WebSocket, context_manager: ContextManager) -> None:
    await websocket.accept()
    call_id: str = ""
    system_prompt = context_manager.get_system_prompt()
    knowledge_string = context_manager.get_knowledge_string()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            interaction_type = msg.get("interaction_type")

            if interaction_type == "call_started":
                call_id = msg.get("call_id", "unknown")
                logger.info("Retell call started: %s", call_id)
                # Send opening greeting
                opening = await _get_claude_response(
                    call_id=call_id,
                    user_text="[CALL STARTED — greet the contact naturally and introduce yourself]",
                    system_prompt=system_prompt,
                    knowledge_string=knowledge_string,
                )
                await _send_response(websocket, opening)

            elif interaction_type in ("response_required", "reminder_required"):
                transcript = msg.get("transcript", [])
                # Get last user utterance
                user_text = _last_user_utterance(transcript)
                if not user_text:
                    user_text = "[silence — ask if they're still there]"

                logger.info("Retell [%s] user said: %s", call_id, user_text[:80])
                reply = await _get_claude_response(
                    call_id=call_id,
                    user_text=user_text,
                    system_prompt=system_prompt,
                    knowledge_string=knowledge_string,
                )
                await _send_response(websocket, reply)

            elif interaction_type == "call_ended":
                call_id = msg.get("call_id", call_id)
                logger.info("Retell call ended: %s", call_id)
                claude_client.clear_history(hash(call_id))
                break

    except WebSocketDisconnect:
        logger.info("Retell WebSocket disconnected (call_id=%s)", call_id)
    except Exception as exc:
        logger.error("Retell WebSocket error: %s", exc)
        try:
            await websocket.close()
        except Exception:
            pass


async def _get_claude_response(
    call_id: str,
    user_text: str,
    system_prompt: str,
    knowledge_string: str,
) -> str:
    # Use call_id hash as chat_id so each call has isolated history
    chat_id = hash(call_id)
    result = claude_client.chat(
        chat_id=chat_id,
        user_message=user_text,
        system_prompt=system_prompt,
        knowledge_string=knowledge_string,
        model=None,  # use default chat model
    )
    if result["success"] and result.get("text"):
        return result["text"]
    logger.error("Claude error during call: %s", result.get("error"))
    return "Desculpe, tive um problema técnico. Pode repetir?"


async def _send_response(websocket: WebSocket, text: str) -> None:
    payload = {
        "response_type": "response",
        "content": text,
        "content_complete": True,
    }
    await websocket.send_text(json.dumps(payload))
    logger.info("Retell response sent: %s", text[:80])


def _last_user_utterance(transcript: list) -> str:
    """Extract the most recent user turn from the transcript."""
    for entry in reversed(transcript):
        if entry.get("role") == "user":
            return entry.get("content", "")
    return ""
