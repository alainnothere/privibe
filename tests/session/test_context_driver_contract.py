"""Context + driver wired together: the payload respects the stored context.

test_frozen_base.py pins that the STORED session survives a binary upgrade;
these tests pin the next seam - ConversationList through the real driver
(OpenAIAdapter.prepare_request) to actual wire bytes. A session is created,
saved, and resumed under a hostile environment (different live tools, flipped
provider config), and the bytes the driver would send must be identical.
This is the seam where the 2026-08-18 cache kill actually happened: not
inside the driver, but in what the caller consulted between the context and
the driver.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from privibe.core.agents.models import AgentProfile, AgentSafety
from privibe.core.config import ProviderConfig, SessionLoggingConfig
from privibe.core.conversation import ConversationList
from privibe.core.llm.backend.generic import OpenAIAdapter
from privibe.core.session.session_loader import BASE_FILENAME, METADATA_FILENAME
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

PROVIDER = ProviderConfig(
    name="llamacpp",
    api_base="http://localhost:8089/v1",
    per_message_reasoning_effort=True,
)


def _tool(name: str) -> AvailableTool:
    return AvailableTool(
        function=AvailableFunction(
            name=name,
            description=f"{name} description",
            parameters={"type": "object", "properties": {}},
        )
    )


TOOLS_V1 = [_tool("read_file")]
TOOLS_V2 = [_tool("read_file"), _tool("tool_added_by_upgrade")]


def _wire_bytes(conv: ConversationList) -> bytes:
    """Exactly what the driver would put on the wire for this context."""
    return (
        OpenAIAdapter()
        .prepare_request(
            model_name="m",
            messages=conv,
            temperature=0.2,
            tools=conv.tools_for_request(),
            max_tokens=None,
            tool_choice=None,
            enable_streaming=False,
            provider=PROVIDER,
            wire_per_message_effort=conv.wire_per_message_effort_for_request(),
        )
        .body
    )


def _make_conversation(
    tools: list[AvailableTool], wire_effort: bool
) -> ConversationList:
    conv = ConversationList(
        tools_provider=lambda: list(tools), wire_effort_provider=lambda: wire_effort
    )
    conv.add(LLMMessage(role=Role.system, content="SYSTEM PROMPT"))
    conv.add(LLMMessage(role=Role.user, content="q1", reasoning_effort="xhigh"))
    conv.add(LLMMessage(role=Role.assistant, content="a1"))
    return conv


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
    return SessionLogger(config, "context-driver-session")


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


def _hostile_resume(session_dir: Path, live_wire_effort: bool) -> ConversationList:
    """Resume under a changed environment: upgraded tools, flipped config."""
    resumed = ConversationList(
        tools_provider=lambda: list(TOOLS_V2),
        wire_effort_provider=lambda: live_wire_effort,
    )
    resumed.restore(session_dir)
    return resumed


# ---------------------------------------------------------------------------
# The seam: stored context -> driver -> identical bytes across resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_produces_byte_identical_wire_payload(
    logger, tool_manager, profile
):
    conv = _make_conversation(TOOLS_V1, wire_effort=True)
    sent = _wire_bytes(conv)  # first request freezes tools + wire flag
    await _save(logger, conv, tool_manager, profile)

    resumed = _hostile_resume(logger.session_dir, live_wire_effort=False)
    assert _wire_bytes(resumed) == sent, (
        "resume re-rendered the conversation differently - the llama.cpp "
        "KV-prefix would fail at the first changed byte and the whole "
        "context would reprocess"
    )


@pytest.mark.asyncio
async def test_growth_after_resume_only_appends(logger, tool_manager, profile):
    conv = _make_conversation(TOOLS_V1, wire_effort=True)
    sent = json.loads(_wire_bytes(conv))
    await _save(logger, conv, tool_manager, profile)

    resumed = _hostile_resume(logger.session_dir, live_wire_effort=False)
    resumed.add(LLMMessage(role=Role.user, content="q2", reasoning_effort="low"))
    grown = json.loads(_wire_bytes(resumed))

    assert {k: v for k, v in grown.items() if k != "messages"} == {
        k: v for k, v in sent.items() if k != "messages"
    }
    assert grown["messages"][: len(sent["messages"])] == sent["messages"]


# ---------------------------------------------------------------------------
# Wire flag: frozen at first use, immune to live config
# ---------------------------------------------------------------------------


def test_wire_flag_freezes_and_ignores_provider_after():
    live = {"on": True}
    conv = ConversationList(wire_effort_provider=lambda: live["on"])
    assert conv.frozen_wire_per_message_effort is None
    assert conv.wire_per_message_effort_for_request() is True

    live["on"] = False  # config "toggles" mid-session
    assert conv.wire_per_message_effort_for_request() is True


def test_wire_flag_without_provider_fails_loudly():
    conv = ConversationList()
    with pytest.raises(RuntimeError, match="no\\s.*provider wired"):
        conv.wire_per_message_effort_for_request()


@pytest.mark.asyncio
async def test_frozen_false_survives_resume_under_flag_on_config(
    logger, tool_manager, profile
):
    conv = _make_conversation(TOOLS_V1, wire_effort=False)
    sent = _wire_bytes(conv)
    assert b"chat_template_kwargs" not in sent
    await _save(logger, conv, tool_manager, profile)

    resumed = _hostile_resume(logger.session_dir, live_wire_effort=True)
    assert resumed.frozen_wire_per_message_effort is False
    assert _wire_bytes(resumed) == sent


# ---------------------------------------------------------------------------
# Legacy sessions: backfill once, then the stored value rules forever
# ---------------------------------------------------------------------------


def _strip_wire_flag_from_disk(session_dir: Path) -> None:
    """Rewrite the session files as an old binary would have written them."""
    for name in (BASE_FILENAME, METADATA_FILENAME):
        path = session_dir / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("per_message_reasoning_effort", None)
        path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_legacy_stamped_session_backfills_true(logger, tool_manager, profile):
    conv = _make_conversation(TOOLS_V1, wire_effort=True)
    _wire_bytes(conv)  # first request freezes tools + wire flag
    await _save(logger, conv, tool_manager, profile)
    _strip_wire_flag_from_disk(logger.session_dir)

    # any(stamped) is what the old driver evaluated, so a stamped legacy
    # session must come back True - reproducing its old rendering exactly.
    resumed = _hostile_resume(logger.session_dir, live_wire_effort=False)
    assert resumed.frozen_wire_per_message_effort is True


@pytest.mark.asyncio
async def test_legacy_unstamped_session_backfills_false_and_never_flips(
    logger, tool_manager, profile
):
    conv = ConversationList(
        tools_provider=lambda: list(TOOLS_V1), wire_effort_provider=lambda: True
    )
    conv.add(LLMMessage(role=Role.system, content="SYSTEM PROMPT"))
    conv.add(LLMMessage(role=Role.user, content="q1"))  # never stamped
    conv.tools_for_request()
    conv._wire_per_message_effort = False  # what the old binary sent: nothing
    await _save(logger, conv, tool_manager, profile)
    _strip_wire_flag_from_disk(logger.session_dir)

    # First resume under the new binary: backfill any(stamped) -> False,
    # persisted to meta.json on the next save.
    resumed = _hostile_resume(logger.session_dir, live_wire_effort=True)
    assert resumed.frozen_wire_per_message_effort is False
    resumed.add(LLMMessage(role=Role.assistant, content="a1"))
    resumed.add(LLMMessage(role=Role.user, content="q2", reasoning_effort="low"))
    await _save(logger, resumed, tool_manager, profile)

    # Second resume: the conversation now CONTAINS a stamp, so a recomputed
    # any(stamped) would flip True and rewrite the system region - the exact
    # first-stamp cache kill. The stored False must rule instead.
    resumed_again = _hostile_resume(logger.session_dir, live_wire_effort=True)
    assert resumed_again.frozen_wire_per_message_effort is False
    assert b"chat_template_kwargs" not in _wire_bytes(resumed_again)
