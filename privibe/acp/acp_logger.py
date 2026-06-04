from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

import time

if TYPE_CHECKING:
    from acp.connection import StreamEvent

ACP_LOG_DIR = Path.home() / ".privibe" / "logs" / "acp"
ACP_LOG_FILE = ACP_LOG_DIR / "messages.jsonl"
MAX_LOG_SIZE_BYTES = 1_000_000
BACKUP_COUNT = 3

ACP_LOGGING_ENABLED_KEY = "VIBE_ACP_LOGGING_ENABLED"

_SESSION_CACHE_TTL = 3600
_SESSION_CACHE_MAXSIZE = 1000
_session_cache: dict[int | str, tuple[str, float]] = {}
_current_session: str | None = None
_logger: logging.Logger | None = None


def is_acp_logging_enabled() -> bool:
    return os.getenv(ACP_LOGGING_ENABLED_KEY, "").lower() in {"1", "true", "yes"}


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(record.msg, separators=(",", ":"))


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    ACP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("acp_messages")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(
        ACP_LOG_FILE,
        maxBytes=MAX_LOG_SIZE_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)

    _logger = logger
    return _logger


def _extract_session_id(message: dict) -> str | None:
    json_str = json.dumps(message)
    match = re.search(r'"(?:session_id|sessionId)":\s*"([^"]+)"', json_str)
    return match.group(1) if match else None


def acp_message_observer(event: StreamEvent) -> None:
    if not is_acp_logging_enabled():
        return

    try:
        global _current_session

        message = event.message
        msg_id = message.get("id", "")

        now = time.monotonic()
        cached = _session_cache.get(msg_id)
        if cached is not None and now - cached[1] < _SESSION_CACHE_TTL:
            session_id = cached[0]
        else:
            if msg_id in _session_cache:
                del _session_cache[msg_id]
            session_id = _extract_session_id(message) or _current_session

        if session_id is not None:
            _current_session = session_id
            if msg_id:
                if len(_session_cache) >= _SESSION_CACHE_MAXSIZE:
                    oldest = min(_session_cache, key=lambda k: _session_cache[k][1])
                    del _session_cache[oldest]
                _session_cache[msg_id] = (session_id, now)

        log_entry: dict = {
            "ts": datetime.now(UTC).isoformat(),
            "dir": "in" if event.direction.value == "incoming" else "out",
            "msg": message,
            **({"session": session_id} if session_id else {}),
        }

        _get_logger().info(log_entry)
    except Exception:
        pass
