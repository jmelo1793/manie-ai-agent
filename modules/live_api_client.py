"""
Generic async HTTP client for fetching live data at query time.
Extend fetch_crm_contact_async() for your specific CRM integration.
"""
import logging
from typing import Any

import httpx

from modules import config

logger = logging.getLogger(__name__)


async def fetch_async(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    params: dict | None = None,
    payload: dict | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """
    Generic async HTTP fetch.

    Returns:
        {'success': True, 'data': any, 'status_code': int}
        {'success': False, 'error': str, 'status_code': int | None}
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                params=params,
                json=payload,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            return {"success": True, "data": data, "status_code": resp.status_code}

    except httpx.HTTPStatusError as exc:
        logger.error("Live API HTTP error %s: %s", exc.response.status_code, exc.response.text)
        return {"success": False, "error": exc.response.text, "status_code": exc.response.status_code}
    except Exception as exc:
        logger.error("Live API request failed: %s", exc)
        return {"success": False, "error": str(exc), "status_code": None}


async def fetch_crm_contact_async(identifier: str) -> dict[str, Any]:
    """
    Look up a contact in the CRM by name, phone, or NIF.
    Configure CRM_API_BASE_URL and CRM_API_TOKEN in .env.

    Returns:
        {'success': True, 'contact': dict}
        {'success': False, 'error': str}
    """
    if not config.CRM_API_BASE_URL:
        logger.debug("CRM_API_BASE_URL not set — skipping live contact lookup")
        return {"success": False, "error": "CRM not configured"}

    result = await fetch_async(
        url=f"{config.CRM_API_BASE_URL}/contacts/search",
        params={"q": identifier},
        headers={"Authorization": f"Bearer {config.CRM_API_TOKEN}"},
    )
    if result["success"]:
        contacts = result["data"]
        contact = contacts[0] if isinstance(contacts, list) and contacts else contacts
        return {"success": True, "contact": contact}
    return {"success": False, "error": result.get("error", "Unknown error")}
