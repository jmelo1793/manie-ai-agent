"""
Retell API client — initiates outbound phone calls.
"""
import logging
from typing import Any

import httpx

from modules import config

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.retellai.com"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {config.RETELL_API_KEY}",
        "Content-Type": "application/json",
    }


async def initiate_call_async(
    to_number: str,
    from_number: str,
    agent_id: str | None = None,
    call_context: str = "",
) -> dict[str, Any]:
    """
    Initiate an outbound call via Retell.
    Retell will connect to our /retell-llm WebSocket for the conversation.

    Returns:
        {'success': True, 'call_id': str}
        {'success': False, 'error': str}
    """
    payload: dict[str, Any] = {
        "from_number": from_number,
        "to_number": to_number,
        "override_agent_id": agent_id or config.RETELL_AGENT_ID,
    }
    if call_context:
        payload["retell_llm_dynamic_variables"] = {"call_context": call_context}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_BASE_URL}/v2/create-phone-call",
                headers=_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            call_id = data.get("call_id", "")
            logger.info("Retell call initiated: call_id=%s to=%s", call_id, to_number)
            return {"success": True, "call_id": call_id}

    except httpx.HTTPStatusError as exc:
        logger.error("Retell HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return {"success": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
    except Exception as exc:
        logger.error("Retell request failed: %s", exc)
        return {"success": False, "error": str(exc)}
