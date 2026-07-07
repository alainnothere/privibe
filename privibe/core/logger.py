from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

from privibe.core.paths import LOG_DIR, LOG_FILE

LOG_DIR.path.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("privibe")

# The stderr handler writes raw log lines to the terminal. That is fine for
# programmatic/ACP runs, but while the Textual TUI owns the screen those writes
# paint straight over the UI (the handler holds the real stderr object, so
# Textual's stream redirect can't intercept it). run_textual_ui suspends it via
# stderr_logging_suspended(); the file handler keeps capturing everything.
_stderr_handler: logging.StreamHandler | None = None


class StructuredLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        ppid = os.getppid()
        pid = os.getpid()
        level = record.levelname
        message = record.getMessage().replace("\\", "\\\\").replace("\n", "\\n")

        line = f"{timestamp} {ppid} {pid} {level} {message}"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info).replace("\n", "\\n")
            line = f"{line} {exc_text}"

        return line


def apply_logging_config(target_logger: logging.Logger) -> logging.StreamHandler:
    """Attach the file + stderr handlers to ``target_logger``.

    Returns the stderr handler so the caller can decide what to do with it (the
    module bootstrap stores it in ``_stderr_handler`` for stderr_logging_suspended;
    tests that configure throwaway loggers simply ignore the return value).
    """
    LOG_DIR.path.mkdir(parents=True, exist_ok=True)

    max_bytes = int(os.environ.get("LOG_MAX_BYTES", 10 * 1024 * 1024))

    if os.environ.get("DEBUG_MODE") == "true":
        log_level_str = "DEBUG"
    else:
        log_level_str = os.environ.get("LOG_LEVEL", "WARNING").upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if log_level_str not in valid_levels:
            log_level_str = "WARNING"

    log_level = getattr(logging, log_level_str, logging.WARNING)

    file_handler = RotatingFileHandler(
        LOG_FILE.path, maxBytes=max_bytes, backupCount=0, encoding="utf-8"
    )
    file_handler.setFormatter(StructuredLogFormatter())
    file_handler.setLevel(log_level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(StructuredLogFormatter())
    stderr_handler.setLevel(log_level)

    # Make sure the logger is not gating logs
    target_logger.setLevel(logging.DEBUG)

    target_logger.addHandler(file_handler)
    target_logger.addHandler(stderr_handler)
    return stderr_handler


@contextmanager
def stderr_logging_suspended() -> Iterator[None]:
    """Detach the stderr log handler for the duration of the block.

    Used while the Textual TUI owns the terminal so log lines can't overpaint
    the UI. The handler is identified by reference (not by type: the file
    handler is also a StreamHandler subclass) and restored on exit, even on
    error. A no-op if the handler was never attached.
    """
    handler = _stderr_handler
    if handler is None or handler not in logger.handlers:
        yield
        return
    logger.removeHandler(handler)
    try:
        yield
    finally:
        logger.addHandler(handler)


_stderr_handler = apply_logging_config(logger)
