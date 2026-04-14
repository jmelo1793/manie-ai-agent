"""
Telegram bot event and command handlers.
Wires together context_manager, claude_client, elevenlabs_client, and aircall_client.
"""
import asyncio
import logging
import re
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from modules import claude_client, config, elevenlabs_client, live_api_client, vapi_client
from modules.context_manager import ContextManager

logger = logging.getLogger(__name__)

# Regex to extract a phone number from a /call command argument
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")


def register_handlers(app: Application, context_manager: ContextManager) -> None:
    app.add_handler(CommandHandler("start", _handle_start))
    app.add_handler(CommandHandler("reset", lambda u, c: _handle_reset(u, c)))
    app.add_handler(CommandHandler("call", lambda u, c: _handle_call_command(u, c, context_manager)))
    app.add_handler(CommandHandler("refresh", lambda u, c: _handle_refresh(u, c, context_manager)))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: _handle_chat(u, c, context_manager))
    )
    logger.info("Telegram handlers registered")


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------

async def _handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Olá! Sou o assistente de IA da Manie.\n\n"
        "Podes fazer-me perguntas sobre clientes e contratos, ou iniciar uma chamada:\n\n"
        "  /call +351912345678 [motivo]\n"
        "  /reset — limpa o histórico de conversa\n"
        "  /refresh — recarrega os ficheiros de contexto"
    )


async def _handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    claude_client.clear_history(chat_id)
    await update.message.reply_text("Histórico de conversa limpo.")


async def _handle_refresh(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    context_manager: ContextManager,
) -> None:
    context_manager.refresh()
    await update.message.reply_text("Contexto recarregado.")


async def _handle_call_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    context_manager: ContextManager,
) -> None:
    chat_id = update.effective_chat.id
    args_text = " ".join(context.args) if context.args else ""

    if not args_text:
        await update.message.reply_text(
            "Uso: /call <número> [motivo]\nExemplo: /call +351912345678 renovação de contrato"
        )
        return

    # Extract phone number
    phone_match = _PHONE_RE.search(args_text)
    if not phone_match:
        await update.message.reply_text(
            "Não consegui encontrar um número de telefone válido. "
            "Exemplo: /call +351912345678 renovação de contrato"
        )
        return

    phone = re.sub(r"[\s\-\(\)]", "", phone_match.group(1))
    purpose = args_text.replace(phone_match.group(1), "").strip() or "assunto não especificado"

    await update.message.chat.send_action(ChatAction.TYPING)
    status_msg = await update.message.reply_text("A gerar script de chamada...")

    # Use the faster haiku model for call script generation
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

    claude_result = claude_client.chat(
        chat_id=chat_id,
        user_message=call_intent,
        system_prompt=system_prompt,
        knowledge_string=knowledge_string,
        model=config.CLAUDE_CALL_MODEL,
    )

    if not claude_result["success"]:
        await status_msg.edit_text(f"Erro ao gerar script: {claude_result.get('error')}")
        return

    tool_call = claude_result.get("tool_call")
    if not tool_call or tool_call["name"] != "initiate_call":
        # Claude replied with text instead of tool use — show it
        await status_msg.edit_text(claude_result.get("text", "Não foi possível gerar o script."))
        return

    tool_input: dict[str, Any] = tool_call["input"]
    script = tool_input.get("call_script", "")
    call_phone = tool_input.get("phone_number", phone)
    first_message = tool_input.get("first_message", "")

    await status_msg.edit_text(f"Script gerado. A sintetizar voz e iniciar chamada para {call_phone}...")

    call_result, audio_result = await _run_call_flow(call_phone, script, tool_call["id"], chat_id, first_message)

    # Build status reply
    lines = []
    if audio_result["success"]:
        lines.append(f"🔊 *Áudio do script:* [ouvir aqui]({audio_result['audio_url']})")
    else:
        lines.append(f"⚠️ Áudio não gerado: {audio_result.get('error')}")

    if call_result["success"]:
        lines.append(f"📞 Chamada iniciada — ID: `{call_result.get('call_id')}`")
    else:
        lines.append(f"❌ Chamada falhou: {call_result.get('error')}")

    await status_msg.edit_text("\n".join(lines), parse_mode="Markdown")


# ------------------------------------------------------------------
# Message handler (chat)
# ------------------------------------------------------------------

async def _handle_chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    context_manager: ContextManager,
) -> None:
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    if not text.strip():
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    # Optionally fetch live CRM data if message looks like a contact lookup
    live_data = ""
    if any(kw in text.lower() for kw in ["cliente", "contrato", "nif", "cpe", "cui", "client", "contact"]):
        crm_result = await live_api_client.fetch_crm_contact_async(text)
        if crm_result["success"]:
            import json
            live_data = json.dumps(crm_result["contact"], ensure_ascii=False, indent=2)

    system_prompt = context_manager.get_system_prompt()
    knowledge_string = context_manager.get_knowledge_string(live_data=live_data)

    result = claude_client.chat(
        chat_id=chat_id,
        user_message=text,
        system_prompt=system_prompt,
        knowledge_string=knowledge_string,
    )

    if not result["success"]:
        await update.message.reply_text(f"Erro: {result.get('error')}")
        return

    tool_call = result.get("tool_call")
    if tool_call and tool_call["name"] == "initiate_call":
        # Claude proactively wants to place a call — confirm with user first
        tool_input = tool_call["input"]
        confirmation = (
            f"Queres que eu ligue para *{tool_input.get('phone_number')}* "
            f"sobre _{tool_input.get('call_purpose')}_?\n\n"
            "Responde com /call para confirmar ou ignora para cancelar."
        )
        await update.message.reply_text(confirmation, parse_mode="Markdown")
        return

    reply_text = result.get("text", "")
    if reply_text:
        await update.message.reply_text(reply_text)


# ------------------------------------------------------------------
# Call flow helper
# ------------------------------------------------------------------

async def _run_call_flow(
    phone: str,
    script: str,
    tool_use_id: str,
    chat_id: int,
    first_message: str = "",
) -> tuple[dict, dict]:
    """
    Initiate a Retell call (Claude handles the live conversation via WebSocket).
    ElevenLabs audio is still generated as a reference script for Telegram.
    Returns (call_result, audio_result).
    """
    audio_task = asyncio.create_task(elevenlabs_client.synthesize_async(script))
    call_task = asyncio.create_task(
        vapi_client.initiate_call_async(
            to_number=phone,
            call_context=script,
            first_message=first_message,
        )
    )
    audio_result, call_result = await asyncio.gather(audio_task, call_task)

    if call_result["success"] and call_result.get("call_id"):
        vapi_client.register_call(call_result["call_id"], chat_id, script)

    status_text = (
        f"Call to {phone}: {'initiated' if call_result['success'] else 'failed'}. "
        f"Script audio: {'ready' if audio_result['success'] else 'failed'}."
    )
    claude_client.submit_tool_result(
        chat_id=chat_id,
        tool_use_id=tool_use_id,
        result_text=status_text,
    )

    return call_result, audio_result
