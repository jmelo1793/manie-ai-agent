"""
Loads client knowledge files from the context/ directory and builds
the strings injected into Claude's system prompt and user messages.
"""
import csv
import json
import logging
from pathlib import Path

import pdfplumber

from modules import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_FILE = "system_prompt.md"


class ContextManager:
    def __init__(self) -> None:
        self._system_prompt: str = ""
        self._knowledge_string: str = ""

    def load(self) -> None:
        """Read all files in context/ and build the in-memory strings."""
        context_dir: Path = config.CONTEXT_DIR

        if not context_dir.exists():
            logger.warning("context/ directory not found at %s", context_dir)
            return

        system_prompt_path = context_dir / _SYSTEM_PROMPT_FILE
        if system_prompt_path.exists():
            self._system_prompt = system_prompt_path.read_text(encoding="utf-8")
            logger.info("Loaded system prompt (%d chars)", len(self._system_prompt))
        else:
            logger.warning("system_prompt.md not found — using empty system prompt")

        parts: list[str] = []
        for path in sorted(context_dir.iterdir()):
            if path.name == _SYSTEM_PROMPT_FILE or path.name.startswith("."):
                continue
            content = self._read_file(path)
            if content:
                parts.append(f"=== FILE: {path.name} ===\n{content}\n=== END: {path.name} ===")

        self._knowledge_string = "\n\n".join(parts)
        logger.info(
            "Loaded %d knowledge file(s) (%d total chars)",
            len(parts),
            len(self._knowledge_string),
        )

    def refresh(self) -> None:
        """Re-read all files from disk."""
        logger.info("Refreshing context...")
        self.load()

    def get_system_prompt(self) -> str:
        return self._system_prompt

    def get_knowledge_string(self, live_data: str = "") -> str:
        """Return concatenated knowledge + optional live data block."""
        if not live_data:
            return self._knowledge_string
        return self._knowledge_string + f"\n\n=== LIVE DATA ===\n{live_data}\n=== END LIVE DATA ==="

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> str:
        try:
            ext = path.suffix.lower()
            if ext == ".pdf":
                return self._read_pdf(path)
            elif ext in {".txt", ".md"}:
                return path.read_text(encoding="utf-8")
            elif ext == ".csv":
                return self._read_csv(path)
            elif ext == ".json":
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                logger.debug("Skipping unsupported file type: %s", path.name)
                return ""
        except Exception as exc:
            logger.error("Failed to read %s: %s", path.name, exc)
            return ""

    @staticmethod
    def _read_pdf(path: Path) -> str:
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)

    @staticmethod
    def _read_csv(path: Path) -> str:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(row.get(h, "")) for h in headers))
        return "\n".join(lines)
