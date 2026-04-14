"""
Vapi API client — initiates and polls outbound phone calls.
"""
import asyncio
import logging
from typing import Any

import httpx

from modules import config

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.vapi.ai"

# call_id → {chat_id, purpose} — used to route end-of-call results
_call_registry: dict[str, dict] = {}


def register_call(call_id: str, chat_id: int, purpose: str) -> None:
    _call_registry[call_id] = {"chat_id": chat_id, "purpose": purpose}


def pop_call(call_id: str) -> dict | None:
    return _call_registry.pop(call_id, None)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.VAPI_API_KEY}",
        "Content-Type": "application/json",
    }


async def initiate_call_async(
    to_number: str,
    call_context: str = "",
    first_message: str = "",
) -> dict[str, Any]:
    """
    Initiate an outbound call via Vapi.

    Returns:
        {'success': True, 'call_id': str}
        {'success': False, 'error': str}
    """
    payload: dict[str, Any] = {
        "phoneNumberId": config.VAPI_PHONE_NUMBER_ID,
        "customer": {"number": to_number},
        "assistantId": config.VAPI_ASSISTANT_ID,
    }
    overrides: dict[str, Any] = {}
    if call_context:
        overrides["variableValues"] = {"call_context": call_context}
    if first_message:
        overrides["firstMessage"] = first_message
    if overrides:
        payload["assistantOverrides"] = overrides

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_BASE_URL}/call/phone",
                headers=_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            call_id = data.get("id", "")
            logger.info("Vapi call initiated: call_id=%s to=%s", call_id, to_number)
            return {"success": True, "call_id": call_id}

    except httpx.HTTPStatusError as exc:
        logger.error("Vapi HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return {"success": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        logger.error("Vapi request failed: %s", exc)
        return {"success": False, "error": str(exc)}


async def get_call(call_id: str) -> dict[str, Any]:
    """
    Fetch current call details from Vapi.

    Returns the raw Vapi call object (status, transcript, artifact, etc.)
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{_BASE_URL}/call/{call_id}",
            headers=_headers(),
        )
        resp.raise_for_status()
        return resp.json()


async def poll_call_until_ended(
    call_id: str,
    poll_interval: int = 5,
    max_wait: int = 3600,
) -> dict[str, Any] | None:
    """
    Poll Vapi every `poll_interval` seconds until the call status is 'ended'.

    Returns the final call object, or None if `max_wait` seconds elapse.
    """
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        try:
            call = await get_call(call_id)
            status = call.get("status", "")
            logger.debug("Vapi poll %s → status=%s", call_id, status)
            if status == "ended":
                return call
        except Exception as exc:
            logger.warning("Vapi poll error for call %s: %s", call_id, exc)

    logger.warning("Vapi poll timed out for call %s after %ds", call_id, max_wait)
    return None
