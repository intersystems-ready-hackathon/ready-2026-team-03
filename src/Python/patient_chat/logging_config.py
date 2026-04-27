import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_FILE = _DEFAULT_LOG_DIR / "patient_chat.log"
_already_configured = False

def setup_logging(level: str | int | None = None) -> None:
    """Configure the root logger once.
    Reads ``LOG_LEVEL`` from the environment (default ``INFO``). Logs go to
    both stderr and a rotating file under ``src/Python/patient_chat/logs/``.
    Safe to call multiple times — only the first call has any effect, which
    matters because Streamlit re-runs the script on every interaction.
    """
    global _already_configured
    if _already_configured:
        return

    resolved_level = level or os.getenv("LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = resolved_level.upper()

    _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # OpenAI / httpx are very chatty at DEBUG; keep them at WARNING unless
    # the user explicitly wants the noise.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _already_configured = True
    logging.getLogger(__name__).info(
        "Logging initialized (level=%s, file=%s)", resolved_level, _LOG_FILE
    )
