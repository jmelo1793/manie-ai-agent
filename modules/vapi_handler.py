"""
Vapi Custom LLM handler.
Vapi sends OpenAI-compatible POST requests; we respond with Claude via SSE stream.
Supports Vapi native tools (dtmf, endCall) by converting between OpenAI and Anthropic formats.

Request format (from Vapi):
  POST /vapi-llm
  {"model": "...", "messages": [...], "stream": true, "call": {...}, "tools": [...]}

Response format (SSE text):
  data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}
  data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
  data: [DONE]

Response format (SSE tool call):
  data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_x", "type": "function", "function": {"name": "dtmf", "arguments": ""}}]}, "finish_reason": null}]}
  data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"dtmf\":\"1\"}"}}]}, "finish_reason": null}]}
  data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
  data: [DONE]
"""
import json
import logging
import uuid
from typing import AsyncGenerator

import anthropic
from fastapi import Request
from fastapi.responses import StreamingResponse

from modules import config
from modules.context_manager import ContextManager

logger = logging.getLogger(__name__)

# Reuse client across requests to avoid reconnection overhead
_anthropic_client = anthropic.AsyncAnthropic(api_key=None)


def _get_client() -> anthropic.AsyncAnthropic:
    global _anthropic_client
    if not _anthropic_client.api_key:
        _anthropic_client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert Vapi's OpenAI-format tool definitions to Anthropic format."""
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool["function"]
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
    return result


def _convert_messages(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-format messages (including tool_calls and tool results)
    to Anthropic format.
    """
    result = []
    for m in messages:
        role = m.get("role")

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            content_text = m.get("content") or ""
            if tool_calls:
                blocks = []
                if content_text:
                    blocks.append({"type": "text", "text": content_text})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""),
                        "input": args,
                    })
                result.append({"role": "assistant", "content": blocks})
            elif content_text:
                result.append({"role": "assistant", "content": content_text})

        elif role == "tool":
            # OpenAI tool result → Anthropic tool_result inside a user message
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": m.get("content", ""),
                }],
            })

        elif role == "user":
            content = m.get("content")
            if content:
                result.append({"role": "user", "content": content})

    return result


async def handle_vapi_llm(request: Request, context_manager: ContextManager) -> StreamingResponse:
    body = await request.json()
    messages: list[dict] = body.get("messages", [])
    call_info: dict = body.get("call", {})
    call_id: str = call_info.get("id", "unknown")
    vapi_tools: list[dict] = body.get("tools", [])

    # Extract call context passed from the /call command
    call_context: str = (
        call_info.get("assistantOverrides", {})
        .get("variableValues", {})
        .get("call_context", "")
    )

    logger.info("Vapi LLM request — call_id=%s messages=%d tools=%d", call_id, len(messages), len(vapi_tools))

    # Build system prompt
    system_prompt = context_manager.get_system_prompt()
    knowledge_string = context_manager.get_knowledge_string()
    full_system = system_prompt
    if knowledge_string:
        full_system += f"\n\n## Knowledge Base\n\n{knowledge_string}"
    if call_context:
        full_system += f"\n\n## This Call\n\n{call_context}"

    claude_messages = _convert_messages(messages)

    if not claude_messages:
        return StreamingResponse(
            _stream_text("Olá, em que posso ajudar?"),
            media_type="text/event-stream",
        )

    anthropic_tools = _openai_tools_to_anthropic(vapi_tools)

    return StreamingResponse(
        _stream_claude(full_system, claude_messages, anthropic_tools),
        media_type="text/event-stream",
    )


async def _stream_claude(
    system: str, messages: list[dict], tools: list[dict]
) -> AsyncGenerator[str, None]:
    client = _get_client()
    kwargs = dict(
        model=config.CLAUDE_CALL_MODEL,
        max_tokens=200,
        system=system,
        messages=messages,
    )
    if tools:
        kwargs["tools"] = tools

    try:
        response = await client.messages.create(**kwargs, stream=False)

        if response.stop_reason == "tool_use":
            # Find the tool_use block and return it in OpenAI streaming format
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
                    args_str = json.dumps(block.input)
                    logger.info("Claude calling tool: %s args=%s", block.name, args_str)

                    # Announce tool call
                    yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': tool_call_id, 'type': 'function', 'function': {'name': block.name, 'arguments': ''}}]}, 'finish_reason': None, 'index': 0}]})}\n\n"
                    # Send arguments
                    yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': args_str}}]}, 'finish_reason': None, 'index': 0}]})}\n\n"
                    break

            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls', 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"

        else:
            # Normal text response — stream word by word for low latency feel
            full_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    full_text += block.text

            # Strip markdown
            clean = (full_text
                     .replace("**", "").replace("*", "")
                     .replace("__", "").replace("_", "")
                     .replace("#", ""))

            if clean:
                chunk = {"choices": [{"delta": {"content": clean}, "finish_reason": None, "index": 0}]}
                yield f"data: {json.dumps(chunk)}\n\n"

            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]})}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.error("Claude error: %s", exc)
        fallback = {"choices": [{"delta": {"content": "Desculpe, pode repetir?"}, "finish_reason": "stop", "index": 0}]}
        yield f"data: {json.dumps(fallback)}\n\n"
        yield "data: [DONE]\n\n"


async def _stream_text(text: str) -> AsyncGenerator[str, None]:
    chunk = {"choices": [{"delta": {"content": text}, "finish_reason": "stop", "index": 0}]}
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
