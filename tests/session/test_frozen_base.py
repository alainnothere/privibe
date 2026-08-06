"""The stone: whatever a session stored is what gets sent, verbatim.

These tests pin the push/pop invariant at the level it is defined: the
payload. A session is created under one "binary" (tools provider), resumed
under another, and the restored payload must be byte-equal to what was
stored - system prompt, tools, and messages - with appends as the only
allowed difference. If a change makes any of these fail, that change is
mutating sent context and must not ship.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from privibe.core.agents.models import AgentProfile, AgentSafety
from privibe.core.config import SessionLoggingConfig
from privibe.core.conversation import ConversationList
from privibe.core.session.session_loader import BASE_FILENAME, SessionLoader
from privibe.core.session.session_logger import SessionLogger
from privibe.core.tools.manager import ToolManager
from privibe.core.types import (
    AgentStats,
    AvailableFunction,
    AvailableTool,
    LLMMessage,
    Role,
)
from tests.conftest import build_test_vibe_config


def _tool(name: str) -> AvailableTool:
    return AvailableTool(
        function=AvailableFunction(
            name=name,
            description=f"{name} description",
            parameters={"type": "object", "properties": {}},
        )
    )


TOOLS_V1 = [_tool("read_file"), _tool("bash")]
TOOLS_V2 = [_tool("read_file"), _tool("bash"), _tool("tool_added_by_upgrade")]
TOOLS_V3 = [_tool("read_file")]


def _payload_fingerprint(conv: ConversationList) -> str:
    """SHA-256 over the prefix as sent: system content + tools JSON.

    Mirrors AgentLoop._check_prefix_integrity so these tests speak the same
    language as the runtime tripwire.
    """
    system_content = ""
    if conv and conv[0].role == Role.system:
        system_content = conv[0].content or ""
    tools_blob = json.dumps(
        [t.model_dump(exclude_none=True) for t in conv.tools_for_request()],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(f"{system_content}\x00{tools_blob}".encode()).hexdigest()


@pytest.fixture
def profile() -> AgentProfile:
    return AgentProfile(
        name="test-agent",
        display_name="Test Agent",
        description="A test agent",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )


@pytest.fixture
def tool_manager() -> ToolManager:
    manager = MagicMock(spec=ToolManager)
    manager.available_tools = {}
    return manager


@pytest.fixture
def logger(tmp_path: Path) -> SessionLogger:
    config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions"), session_prefix="test", enabled=True
    )
    return SessionLogger(config, "frozen-base-session")


def _make_conversation(tools: list[AvailableTool]) -> ConversationList:
    conv = ConversationList(tools_provider=lambda: list(tools))
    conv.add(LLMMessage(role=Role.system, content="SYSTEM PROMPT v1"))
    conv.add(LLMMessage(role=Role.user, content="hello"))
    return conv


async def _save(
    logger: SessionLogger, conv: ConversationList, tool_manager, profile
) -> None:
    await logger.save_interaction(
        conv,
        AgentStats(),
        build_test_vibe_config(active_model="test-model", models=[], providers=[]),
        tool_manager,
        profile,
    )


# ---------------------------------------------------------------------------
# ConversationList tools ownership
# ---------------------------------------------------------------------------


def test_tools_freeze_on_first_use_and_ignore_provider_after():
    live = {"tools": TOOLS_V1}
    conv = ConversationList(tools_provider=lambda: list(live["tools"]))
    assert conv.frozen_tools is None
    first = conv.tools_for_request()
    assert first == TOOLS_V1

    live["tools"] = TOOLS_V2  # binary "upgrades" mid-session
    assert conv.tools_for_request() == TOOLS_V1


def test_freeze_tools_twice_fails_loudly():
    conv = ConversationList()
    conv.freeze_tools(TOOLS_V1)
    with pytest.raises(ValueError, match="already frozen"):
        conv.freeze_tools(TOOLS_V2)


def test_tools_for_request_without_provider_or_freeze_fails_loudly():
    conv = ConversationList()
    with pytest.raises(RuntimeError, match="no tools provider"):
        conv.tools_for_request()


# ---------------------------------------------------------------------------
# The stone: stored payload survives a binary upgrade byte-for-byte
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restored_payload_is_byte_identical_across_upgrade(
    logger, tool_manager, profile
):
    conv = _make_conversation(TOOLS_V1)
    conv.tools_for_request()  # first request freezes
    stored_fingerprint = _payload_fingerprint(conv)
    stored_messages = [m.model_dump(exclude_none=True) for m in conv]
    await _save(logger, conv, tool_manager, profile)
    assert (logger.session_dir / BASE_FILENAME).exists()

    # "New binary": provider yields different tools. No config_getter, so
    # restore appends nothing and the comparison is exact.
    resumed = ConversationList(tools_provider=lambda: list(TOOLS_V2))
    resumed.restore(logger.session_dir)

    assert _payload_fingerprint(resumed) == stored_fingerprint
    resumed_messages = [m.model_dump(exclude_none=True) for m in resumed]
    assert resumed_messages == stored_messages


@pytest.mark.asyncio
async def test_restore_reads_tools_from_base_json_not_live_code(
    logger, tool_manager, profile
):
    conv = _make_conversation(TOOLS_V1)
    conv.tools_for_request()
    await _save(logger, conv, tool_manager, profile)

    resumed = ConversationList(tools_provider=lambda: list(TOOLS_V2))
    resumed.restore(logger.session_dir)
    assert resumed.frozen_tools == TOOLS_V1
    assert resumed.tools_for_request() == TOOLS_V1


# ---------------------------------------------------------------------------
# Migration kludge: pre-base.json sessions adopt once, then freeze forever
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_old_session_adopts_live_tools_once_then_frozen(
    logger, tool_manager, profile
):
    # Phase 1: a session saved by the "old binary" - no base.json.
    conv = _make_conversation(TOOLS_V1)
    conv.tools_for_request()
    await _save(logger, conv, tool_manager, profile)
    (logger.session_dir / BASE_FILENAME).unlink()

    # Phase 2: first resume under the "new binary" (TOOLS_V2). The kludge
    # leaves tools unfrozen; first request adopts live; next save persists.
    resumed = ConversationList(tools_provider=lambda: list(TOOLS_V2))
    resumed.restore(logger.session_dir)
    assert resumed.frozen_tools is None
    assert resumed[0].content == "SYSTEM PROMPT v1"  # meta.json fallback
    assert resumed.tools_for_request() == TOOLS_V2
    resumed.add(LLMMessage(role=Role.assistant, content="hi"))
    await _save(logger, resumed, tool_manager, profile)
    assert (logger.session_dir / BASE_FILENAME).exists()

    # Phase 3: another upgrade (TOOLS_V3). The adopted set wins forever.
    resumed_again = ConversationList(tools_provider=lambda: list(TOOLS_V3))
    resumed_again.restore(logger.session_dir)
    assert resumed_again.frozen_tools == TOOLS_V2
    assert resumed_again.tools_for_request() == TOOLS_V2


# ---------------------------------------------------------------------------
# base.json is write-once by construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_base_never_overwrites(tmp_path: Path):
    session_dir = tmp_path / "s"
    session_dir.mkdir()
    original = {"system_prompt": {"role": "system", "content": "A"}, "tools": []}
    await SessionLogger.persist_base(original, session_dir)

    attacker = {"system_prompt": {"role": "system", "content": "B"}, "tools": []}
    await SessionLogger.persist_base(attacker, session_dir)

    on_disk = json.loads((session_dir / BASE_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == original


@pytest.mark.asyncio
async def test_save_interaction_does_not_rewrite_existing_base(
    logger, tool_manager, profile
):
    conv = _make_conversation(TOOLS_V1)
    conv.tools_for_request()
    await _save(logger, conv, tool_manager, profile)
    base_path = logger.session_dir / BASE_FILENAME
    sentinel = base_path.read_bytes()

    conv.add(LLMMessage(role=Role.assistant, content="turn 2"))
    await _save(logger, conv, tool_manager, profile)
    assert base_path.read_bytes() == sentinel


def test_load_base_missing_returns_none(tmp_path: Path):
    assert SessionLoader.load_base(tmp_path) is None


def test_load_base_corrupt_raises_instead_of_silent_fallback(tmp_path: Path):
    (tmp_path / BASE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="session base"):
        SessionLoader.load_base(tmp_path)
