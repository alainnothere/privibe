"""DEBUG LLM COMMUNICATIONS — shared payload dumper.

Both the Anthropic and OpenAI-style adapters call into this when
`llm_debug_dump` is on, so we have a single source of truth for filename
format and on-disk layout.

Filenames carry enough context to compare turns from the same session
without ambiguity:

    {YYYYMMDD-HHMMSS-mmm}_{seq}_{session8}_{kind}_payload_{nmsgs}msgs.json

  - timestamp + monotonic sequence guarantees ordering across rapid turns
    even when filesystem mtime rounds to the second
  - session8 is the first 8 chars of agent_loop.session_id so dumps from
    different privibe instances don't collide on the same turn number
  - kind is "real" / "warmup" / "compaction" / "count_tokens" so filtering
    is trivial after the fact
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import threading
from pathlib import Path
from typing import Any

# Module-level state populated by the agent loop before each LLM call.
_enabled: bool = False
_session_id: str | None = None
_kind: str = "real"
_seq = itertools.count(1)
_lock = threading.Lock()


def set_debug_dump_payload(enabled: bool) -> None:
    """Compatibility shim — agent_loop calls this each turn."""
    global _enabled
    _enabled = enabled


def set_dump_context(*, session_id: str | None, kind: str = "real") -> None:
    """Update the session-id + kind tag the next dump will carry."""
    global _session_id, _kind
    _session_id = session_id
    _kind = kind or "real"


def is_enabled() -> bool:
    return _enabled


def _short_session(sid: str | None) -> str:
    if not sid:
        return "nosess"
    return sid.replace("-", "")[:8]


def _dump_dir() -> Path:
    """Resolve the debug dump directory lazily so VIBE_HOME overrides apply."""
    # Lazy import — paths module pulls privibe package init, which is
    # fine at runtime but we don't want it eagerly at module import time.
    from privibe.core.paths import DEBUG_DIR  # noqa: PLC0415
    return DEBUG_DIR.path


def dump_payload(body: bytes | str, payload: dict[str, Any] | None = None) -> Path | None:
    """Write the serialized request body. Returns the path, or None if disabled.

    `payload` is the un-serialized dict — we pull len(messages) out of it
    for the filename so dumps can be sorted by message count at a glance.
    """
    if not _enabled:
        return None
    try:
        n_msgs = len(payload.get("messages", [])) if payload else 0
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ms precision
        with _lock:
            seq = next(_seq)
        fname = (
            f"{ts}_{seq:04d}_{_short_session(_session_id)}_{_kind}"
            f"_payload_{n_msgs}msgs.json"
        )
        d = _dump_dir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / fname
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        p.write_text(text, encoding="utf-8")
        return p
    except Exception:
        # Debug dump must never break the request path. Swallow.
        return None
