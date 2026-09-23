"""MessageMeta and the immutability law of stored messages.

Two things are pinned here, and they are the same law:

1. Nothing taken out of a ConversationList can be edited through the
   reference. `frozen=True` on LLMMessage only guards the message's own
   attributes; the pieces the chat template renders into the KV-cache prefix
   (tool calls) must be frozen and un-appendable themselves.
2. `meta` is display-only. It can never reach a backend payload (field-level
   exclude, so no model_dump anywhere emits it), it survives the session log
   round trip, it survives a rewind, and it cannot grow without editing the
   type and the field list below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from privibe.core.agents.models import AgentProfile, AgentSafety
from privibe.core.config import SessionLoggingConfig
from privibe.core.conversation import ConversationList
from privibe.core.session.session_loader import SessionLoader
from privibe.core.session.session_logger import SessionLogger, stored_message_dict
from privibe.core.tools.manager import ToolManager
from privibe.core.types import (
    AgentStats,
    FileDiff,
    FunctionCall,
    LLMMessage,
    MessageMeta,
    Role,
    ServerTimings,
    ToolCall,
    ToolOutcome,
)
from tests.conftest import build_test_vibe_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The complete field list. Adding a field to MessageMeta means adding it here
# on purpose; see the class docstring for the rule on what qualifies.
META_FIELDS = {
    "submitted_at",
    "request_sent_at",
    "first_chunk_at",
    "response_done_at",
    "server_timings",
    "approval_asked_at",
    "approval_answered_at",
    "tool_started_at",
    "tool_finished_at",
    "tool_outcome",
    "tool_result",
    "file_diff",
}


def _full_meta() -> MessageMeta:
    return MessageMeta(
        submitted_at=1.0,
        request_sent_at=2.0,
        first_chunk_at=2.5,
        response_done_at=3.0,
        server_timings=ServerTimings(
            cache_n=100, prompt_n=20, prompt_ms=50.0, predicted_n=10, predicted_ms=80.0
        ),
        approval_asked_at=4.0,
        approval_answered_at=5.0,
        tool_started_at=6.0,
        tool_finished_at=7.5,
        tool_outcome=ToolOutcome.success,
        tool_result={"command": "ls", "stdout": "a\nb", "stderr": "", "returncode": 0},
        file_diff=FileDiff(path="x.py", kind="diff", hunks=[[("diff-added", "+x")]]),
    )


def _call() -> ToolCall:
    return ToolCall(
        id="call-1",
        index=0,
        function=FunctionCall(name="bash", arguments='{"command": "ls"}'),
    )


def _conversation(with_meta: bool) -> ConversationList:
    meta = _full_meta() if with_meta else None
    conv = ConversationList()
    conv.add(LLMMessage(role=Role.system, content="sys"))
    conv.add(LLMMessage(role=Role.user, content="hi", meta=meta))
    conv.add(
        LLMMessage(role=Role.assistant, content="", tool_calls=[_call()], meta=meta)
    )
    conv.add(
        LLMMessage(
            role=Role.tool,
            name="bash",
            tool_call_id="call-1",
            content="command: ls\nstdout: a\nb\nstderr: \nreturncode: 0",
            meta=meta,
        )
    )
    conv.add(LLMMessage(role=Role.assistant, content="done", meta=meta))
    return conv


# ---------------------------------------------------------------------------
# 1. The mutation law
# ---------------------------------------------------------------------------


class TestStoredMessagesCannotBeEditedThroughAReference:
    def test_message_attributes(self) -> None:
        msg = _conversation(with_meta=True)[2]
        with pytest.raises(ValidationError):
            msg.content = "x"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            msg.meta = None  # type: ignore[misc]

    def test_tool_calls_cannot_be_appended(self) -> None:
        msg = _conversation(with_meta=True)[2]
        assert msg.tool_calls is not None
        with pytest.raises(AttributeError):
            msg.tool_calls.append(_call())  # type: ignore[attr-defined]

    def test_tool_call_pieces_are_frozen(self) -> None:
        msg = _conversation(with_meta=True)[2]
        assert msg.tool_calls is not None
        call = msg.tool_calls[0]
        with pytest.raises(ValidationError):
            call.id = "z"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            call.function.arguments = '{"command": "rm -rf /"}'  # type: ignore[misc]
        with pytest.raises(ValidationError):
            call.function.name = "other"  # type: ignore[misc]

    def test_meta_and_its_pieces_are_frozen(self) -> None:
        msg = _conversation(with_meta=True)[3]
        assert msg.meta is not None
        with pytest.raises(ValidationError):
            msg.meta.tool_started_at = 0.0  # type: ignore[misc]
        assert msg.meta.server_timings is not None
        with pytest.raises(ValidationError):
            msg.meta.server_timings.cache_n = 0  # type: ignore[misc]
        assert msg.meta.file_diff is not None
        with pytest.raises(ValidationError):
            msg.meta.file_diff.kind = "sample"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            msg.meta.file_diff.hunks.append(())  # type: ignore[attr-defined]

    def test_lists_still_validate_into_the_tuple(self) -> None:
        msg = LLMMessage(role=Role.assistant, tool_calls=[_call()])
        assert isinstance(msg.tool_calls, tuple)
        diff = FileDiff(path="x", kind="diff", hunks=[[("a", "b")]])
        assert diff.hunks == ((("a", "b"),),)


# ---------------------------------------------------------------------------
# 2. meta never reaches a dump; the field list is closed
# ---------------------------------------------------------------------------


class TestMetaIsInvisibleToEveryDump:
    def test_excluded_even_when_explicitly_included(self) -> None:
        msg = LLMMessage(role=Role.user, content="hi", meta=_full_meta())
        assert "meta" not in msg.model_dump()
        assert "meta" not in msg.model_dump(exclude_none=True)
        assert "meta" not in msg.model_dump(include={"role", "meta"})
        assert "meta" not in msg.model_dump_json()

    def test_field_list_is_exactly_this(self) -> None:
        assert set(MessageMeta.model_fields) == META_FIELDS

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageMeta(gap_seconds=1.0)  # type: ignore[call-arg]

    def test_server_timings_ignores_fields_we_do_not_read(self) -> None:
        t = ServerTimings.model_validate({
            "cache_n": 5,
            "prompt_per_second": 32.3,
            "brand_new": 1,
        })
        assert t.cache_n == 5
        assert "brand_new" not in t.model_dump()

    def test_accumulator_carries_meta(self) -> None:
        a = LLMMessage(role=Role.assistant, content="a")
        b = LLMMessage(role=Role.assistant, content="b", meta=_full_meta())
        assert (a + b).meta == _full_meta()
        assert (b + a).meta == _full_meta()


# ---------------------------------------------------------------------------
# 3. Storage round trip and rewind
# ---------------------------------------------------------------------------


def _logger(tmp_path: Path, session_id: str) -> SessionLogger:
    config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions"), session_prefix="test", enabled=True
    )
    return SessionLogger(config, session_id)


async def _save(logger: SessionLogger, conv: ConversationList) -> Path:
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    profile = AgentProfile(
        name="t",
        display_name="T",
        description="t",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )
    await logger.save_interaction(
        messages=conv,
        stats=AgentStats(),
        base_config=build_test_vibe_config(
            active_model="test-model", models=[], providers=[]
        ),
        tool_manager=tool_manager,
        agent_profile=profile,
    )
    assert logger.session_dir is not None
    return logger.session_dir


def test_stored_dict_is_the_wire_dict_plus_meta() -> None:
    msg = LLMMessage(role=Role.user, content="hi", meta=_full_meta())
    stored = stored_message_dict(msg)
    assert stored["meta"] == _full_meta().model_dump(exclude_none=True)
    without = dict(stored)
    del without["meta"]
    assert without == msg.model_dump(exclude_none=True)


def test_stored_dict_without_meta_has_no_meta_key() -> None:
    assert "meta" not in stored_message_dict(LLMMessage(role=Role.user, content="hi"))


@pytest.mark.asyncio
async def test_meta_round_trips_through_the_session_log(tmp_path: Path) -> None:
    conv = _conversation(with_meta=True)
    session_dir = await _save(_logger(tmp_path, "s1"), conv)

    restored, _ = SessionLoader.load_session(session_dir)

    assert len(restored) == 4
    for original, back in zip(conv[1:], restored, strict=True):
        assert back.meta == original.meta
        assert back.meta == _full_meta()
        # everything else is what the wire would have seen
        assert back.model_dump(exclude_none=True) == original.model_dump(
            exclude_none=True
        )


@pytest.mark.asyncio
async def test_messages_without_meta_restore_without_meta(tmp_path: Path) -> None:
    session_dir = await _save(_logger(tmp_path, "s2"), _conversation(with_meta=False))
    restored, _ = SessionLoader.load_session(session_dir)
    assert all(m.meta is None for m in restored)


@pytest.mark.asyncio
async def test_rewind_fork_keeps_meta_on_survivors_only(tmp_path: Path) -> None:
    conv = _conversation(with_meta=True)
    logger = _logger(tmp_path, "s3")
    first_dir = await _save(logger, conv)

    # Rewind drops the tool round trip and the final reply, then forks to
    # a fresh session (what RewindManager does after truncation).
    conv.rewind(3)
    logger.reset_session("s4")
    forked_dir = await _save(logger, conv)

    assert forked_dir != first_dir
    survivors, _ = SessionLoader.load_session(forked_dir)
    assert [m.role for m in survivors] == [Role.user]
    assert survivors[0].meta == _full_meta()
    # the original session is untouched: all four still there, with meta
    originals, _ = SessionLoader.load_session(first_dir)
    assert len(originals) == 4
    assert all(m.meta == _full_meta() for m in originals)


def test_restore_puts_meta_back_on_the_conversation(tmp_path: Path) -> None:
    """ConversationList.restore is the resume path; it must see meta."""
    import asyncio

    conv = _conversation(with_meta=True)
    session_dir = asyncio.run(_save(_logger(tmp_path, "s5"), conv))

    fresh = ConversationList(config_getter=lambda: build_test_vibe_config())
    fresh.restore(session_dir)

    by_role: dict[Any, list[LLMMessage]] = {}
    for m in fresh:
        by_role.setdefault(m.role, []).append(m)
    assert by_role[Role.tool][0].meta == _full_meta()
    assert by_role[Role.user][0].meta == _full_meta()
