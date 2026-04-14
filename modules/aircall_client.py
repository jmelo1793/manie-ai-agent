"""
Aircall REST API client — outbound call initiation.

Note on audio injection:
  Aircall webhooks are informational (no TwiML-style response control).
  V1 strategy: initiate the call, post the MP3 audio URL to Telegram so the
  agent can hear the script. The /aircall/webhook endpoint in main.py is
  scaffolded for a future Twilio bridge integration.
"""
import logging
from typing import Any

import httpx

from modules import config

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.aircall.io/v1"


def _auth() -> tuple[str, str]:
    return (config.AIRCALL_API_ID, config.AIRCALL_API_TOKEN)


async def initiate_call_async(
    to_number: str,
    number_id: int | None = None,
    external_key: str = "",
) -> dict[str, Any]:
    """
    Initiate an outbound call via Aircall.
    Endpoint: POST /v1/numbers/{number_id}/dial

    Returns:
        {'success': True, 'call_id': str, 'status': str}
        {'success': False, 'error': str}
    """
    nid = number_id or config.AIRCALL_NUMBER_ID
    user_id = config.AIRCALL_USER_ID
    payload: dict[str, Any] = {"to": to_number, "number_id": nid}
    if user_id:
        payload["user_id"] = int(user_id)
    if external_key:
        payload["external_key"] = external_key

    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=15) as client:
            resp = await client.post(f"{_BASE_URL}/calls", json=payload)
            resp.raise_for_status()
            data = resp.json()
            call = data.get("call", {})
            call_id = str(call.get("id", ""))
            status = call.get("status", "unknown")
            logger.info("Aircall initiated: call_id=%s to=%s status=%s", call_id, to_number, status)
            return {"success": True, "call_id": call_id, "status": status}

    except httpx.HTTPStatusError as exc:
        logger.error("Aircall HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return {"success": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        logger.error("Aircall request failed: %s", exc)
        return {"success": False, "error": str(exc)}


async def get_call_status_async(call_id: str) -> dict[str, Any]:
    """
    Fetch the current status of a call.

    Returns:
        {'success': True, 'status': str, 'duration': int}
        {'success': False, 'error': str}
    """
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=10) as client:
            resp = await client.get(f"{_BASE_URL}/calls/{call_id}")
            resp.raise_for_status()
            call = resp.json().get("call", {})
            return {
                "success": True,
                "status": call.get("status", "unknown"),
                "duration": call.get("duration", 0),
            }
    except Exception as exc:
        logger.error("Aircall get_call_status failed: %s", exc)
        return {"success": False, "error": str(exc)}
