"""
ElevenLabs text-to-speech client.
Uses streaming synthesis to minimize time-to-file.
"""
import logging
import uuid
from pathlib import Path
from typing import Any

from elevenlabs.client import ElevenLabs

from modules import config

logger = logging.getLogger(__name__)


def _get_client() -> ElevenLabs:
    return ElevenLabs(api_key=config.ELEVENLABS_API_KEY)


async def synthesize_async(text: str, voice_id: str | None = None) -> dict[str, Any]:
    """
    Convert text to speech and save as MP3 in audio_cache/.

    Returns:
        {'success': True, 'audio_path': Path, 'audio_url': str}
        {'success': False, 'error': str}
    """
    chosen_voice = voice_id or config.ELEVENLABS_VOICE_ID
    audio_id = uuid.uuid4().hex
    output_path: Path = config.AUDIO_CACHE_DIR / f"{audio_id}.mp3"
    config.AUDIO_CACHE_DIR.mkdir(exist_ok=True)

    try:
        client = _get_client()
        # generate() returns a generator of audio bytes (streaming)
        audio_stream = client.text_to_speech.convert(
            voice_id=chosen_voice,
            text=text,
            model_id=config.ELEVENLABS_MODEL_ID,
            output_format="mp3_44100_128",
        )
        with output_path.open("wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        audio_url = f"{config.APP_BASE_URL}/audio/{audio_id}.mp3"
        logger.info("Synthesized audio: %s (%d bytes)", output_path.name, output_path.stat().st_size)
        return {"success": True, "audio_path": output_path, "audio_url": audio_url}

    except Exception as exc:
        logger.error("ElevenLabs synthesis failed: %s", exc)
        return {"success": False, "error": str(exc)}


def delete_audio(audio_path: Path) -> None:
    """Remove a cached audio file after the call ends."""
    try:
        if audio_path.exists():
            audio_path.unlink()
            logger.info("Deleted audio file: %s", audio_path.name)
    except Exception as exc:
        logger.warning("Could not delete audio file %s: %s", audio_path, exc)
