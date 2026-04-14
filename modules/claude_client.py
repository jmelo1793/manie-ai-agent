"""
Anthropic Claude API wrapper.
Maintains per-user conversation history and handles tool use round-trips.
"""
import logging
from typing import Any

import anthropic

from modules import config

logger = logging.getLogger(__name__)

# In-memory conversation history keyed by Telegram chat_id
_histories: dict[int, list[dict]] = {}
MAX_HISTORY_MESSAGES = 20  # 10 turns

TOOLS: list[dict] = [
    {
        "name": "initiate_call",
        "description": (
            "Initiate an outbound phone call with a synthesized voice script. "
            "Use this when the user asks to call someone. "
            "Generate the full call_script inside this tool call — it will be spoken aloud."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "phone_number": {
                    "type": "string",
                    "description": "Phone number in E.164 format, e.g. +351912345678",
                },
                "call_purpose": {
                    "type": "string",
                    "description": "Brief description of the call topic",
                },
                "contact_name": {
                    "type": "string",
                    "description": "Name of the person being called (if known)",
                },
                "call_script": {
                    "type": "string",
                    "description": (
                        "Full natural-language script to be spoken during the call. "
                        "Write it as spoken dialogue, warm and professional."
                    ),
                },
                "first_message": {
                    "type": "string",
                    "description": (
                        "The very first sentence spoken when the call connects. "
                        "One sentence only. Greeting + Manie + direct reason. No fluff. "
                        "Example: 'Bom dia, sou a Marta da Ménie, ligo para confirmar o estado do contrato do cliente João Silva.'"
                    ),
                },
            },
            "required": ["phone_number", "call_purpose", "call_script", "first_message"],
        },
    }
]


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _trim_history(chat_id: int) -> None:
    history = _histories.get(chat_id, [])
    if len(history) > MAX_HISTORY_MESSAGES:
        _histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]


def _has_tool_use(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        (hasattr(b, "type") and b.type == "tool_use")
        or (isinstance(b, dict) and b.get("type") == "tool_use")
        for b in content
    )


def _has_tool_result(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


def _clean_dangling_tool_use(chat_id: int) -> None:
    """
    Remove any incomplete tool_use sequences at the end of history.
    Claude requires every tool_use to be immediately followed by a tool_result.
    If they're missing (e.g. due to a failed previous call), we strip them out.
    """
    history = _histories.get(chat_id, [])
    if not history:
        return

    last = history[-1]

    # Case 1: history ends with assistant tool_use (no tool_result after it)
    if last["role"] == "assistant" and _has_tool_use(last.get("content", [])):
        history.pop()
        if history and history[-1]["role"] == "user":
            history.pop()
        logger.warning("Cleaned dangling tool_use from history (chat_id=%s)", chat_id)

    # Case 2: history ends with user tool_result (no final assistant response)
    elif last["role"] == "user" and _has_tool_result(last.get("content", [])):
        history.pop()
        if history and history[-1]["role"] == "assistant" and _has_tool_use(history[-1].get("content", [])):
            history.pop()
        if history and history[-1]["role"] == "user":
            history.pop()
        logger.warning("Cleaned orphan tool_result from history (chat_id=%s)", chat_id)


def chat(
    chat_id: int,
    user_message: str,
    system_prompt: str,
    knowledge_string: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Send a message to Claude and return the response.

    Returns:
        {'success': True, 'text': str, 'tool_call': dict | None}
        {'success': False, 'error': str}
    """
    chosen_model = model or config.CLAUDE_CHAT_MODEL

    # Build the full system prompt
    full_system = system_prompt
    if knowledge_string:
        full_system += f"\n\n## Knowledge Base\n\n{knowledge_string}"

    # Ensure history is in a valid state before appending
    if chat_id not in _histories:
        _histories[chat_id] = []
    _clean_dangling_tool_use(chat_id)
    _histories[chat_id].append({"role": "user", "content": user_message})
    _trim_history(chat_id)

    try:
        client = _get_client()
        response = client.messages.create(
            model=chosen_model,
            max_tokens=2048,
            system=full_system,
            tools=TOOLS,
            messages=_histories[chat_id],
        )

        # Append assistant response to history
        _histories[chat_id].append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if tool_block:
                logger.info("Claude requested tool: %s", tool_block.name)
                return {
                    "success": True,
                    "text": "",
                    "tool_call": {
                        "id": tool_block.id,
                        "name": tool_block.name,
                        "input": tool_block.input,
                    },
                }

        # Normal text response
        text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )
        return {"success": True, "text": text, "tool_call": None}

    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", exc)
        return {"success": False, "error": str(exc)}


def submit_tool_result(
    chat_id: int,
    tool_use_id: str,
    result_text: str,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Submit a tool result back to Claude and get the final text response.

    Returns:
        {'success': True, 'text': str}
        {'success': False, 'error': str}
    """
    chosen_model = model or config.CLAUDE_CHAT_MODEL

    if chat_id not in _histories:
        return {"success": False, "error": "No history for this chat"}

    tool_result_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
            }
        ],
    }
    _histories[chat_id].append(tool_result_msg)

    try:
        client = _get_client()
        response = client.messages.create(
            model=chosen_model,
            max_tokens=1024,
            tools=TOOLS,
            messages=_histories[chat_id],
        )
        _histories[chat_id].append({"role": "assistant", "content": response.content})

        text = next(
            (b.text for b in response.content if hasattr(b, "text")), ""
        )
        return {"success": True, "text": text}

    except anthropic.APIError as exc:
        logger.error("Claude API error (tool result): %s", exc)
        # Rollback the tool_result so history stays clean
        if _histories.get(chat_id) and _histories[chat_id][-1] is tool_result_msg:
            _histories[chat_id].pop()
        return {"success": False, "error": str(exc)}


def clear_history(chat_id: int) -> None:
    _histories.pop(chat_id, None)
    logger.info("Cleared history for chat_id=%s", chat_id)
