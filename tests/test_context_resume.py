"""A resumed conversation knows how big it is before anything is sent.

meta.json has carried the stats block (context_tokens included) on every
save for as long as there has been a meta.json; nothing read it back. A
resumed loop reporting zero context let a full conversation be sent to a
server that refused it, turn after turn, with auto-compact never firing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from privibe.cli.context_report import (
    OVERFLOW_ADVICE,
    build_context_report,
    format_context_report,
    run_context_query,
)
from privibe.cli.entrypoint import parse_arguments
from privibe.core.config import SessionLoggingConfig
from privibe.core.llm.exceptions import BackendError, PayloadSummary
from privibe.core.middleware import MiddlewareAction, MiddlewareResult
from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import AgentStats, AssistantEvent
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)


def _write_session(
    root: Path, session_id: str = "abc12345-0000-0000-0000-000000000000",
    stats: dict[str, Any] | None = None,
) -> Path:
    session_dir = root / f"session_{session_id}"
    session_dir.mkdir(parents=True)
    lines = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    (session_dir / "messages.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    meta: dict[str, Any] = {
        "session_id": session_id,
        "start_time": "2026-11-06T20:02:00+00:00",
        "end_time": "2026-11-06T20:02:30+00:00",
        "git_commit": None,
        "git_branch": None,
        "environment": {},
        "username": "u",
        "name": "big one",
    }
    if stats is not None:
        meta["stats"] = stats
    (session_dir / "meta.json").write_text(json.dumps(meta))
    return session_dir


# ── loader + stats ────────────────────────────────────────────────────────


def test_load_saved_stats_reads_the_block(tmp_path: Path) -> None:
    d = _write_session(tmp_path, stats={"context_tokens": 120_000, "steps": 7})
    assert SessionLoader.load_saved_stats(d) == {"context_tokens": 120_000, "steps": 7}


def test_load_saved_stats_none_when_absent_or_broken(tmp_path: Path) -> None:
    assert SessionLoader.load_saved_stats(_write_session(tmp_path, "a")) is None
    assert SessionLoader.load_saved_stats(tmp_path / "nope") is None
    d = _write_session(tmp_path, "b")
    (d / "meta.json").write_text("{not json")
    assert SessionLoader.load_saved_stats(d) is None


def test_restore_saved_fires_context_listener_last_with_totals_in_place() -> None:
    stats = AgentStats()
    seen: list[tuple[int, int]] = []
    stats.add_listener(
        "context_tokens", lambda s: seen.append((s.context_tokens, s.session_prompt_tokens))
    )
    stats.restore_saved(
        {
            "context_tokens": 120_000,
            "session_prompt_tokens": 9_000_000,
            "session_completion_tokens": 50_000,
            "steps": 42,
            "tool_calls_succeeded": 9,
            "tokens_per_second": 55.0,  # not restored: this process never ran that turn
            "input_price_per_million": 3.0,  # not restored: pricing follows the live model
            "session_cost": "garbage",
            "tool_calls_failed": True,  # bool is not a count
        }
    )
    assert seen == [(120_000, 9_000_000)]
    assert stats.steps == 42
    assert stats.session_completion_tokens == 50_000
    assert stats.tool_calls_succeeded == 9
    assert stats.tokens_per_second == 0.0
    assert stats.input_price_per_million == 0.0
    assert stats.tool_calls_failed == 0


# ── agent loop ────────────────────────────────────────────────────────────


def _logging_config(tmp_path: Path, **kwargs: Any):
    return build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path), session_prefix="session"
        ),
        **kwargs,
    )


def test_adopt_saved_session_restores_messages_and_context(tmp_path: Path) -> None:
    sid = "abc12345-0000-0000-0000-000000000000"
    d = _write_session(tmp_path, sid, stats={"context_tokens": 120_000, "steps": 3})
    agent = build_test_agent_loop(config=_logging_config(tmp_path))
    assert agent.stats.context_tokens == 0

    agent.adopt_saved_session(sid, d)

    assert agent.session_id == sid
    assert agent.stats.context_tokens == 120_000
    assert agent.stats.steps == 3
    assert [m.content for m in agent.messages if m.role.value == "user"][0] == "hello"


def test_restore_saved_stats_without_stats_leaves_zero(tmp_path: Path) -> None:
    sid = "abc12345-0000-0000-0000-000000000000"
    d = _write_session(tmp_path, sid)
    agent = build_test_agent_loop(config=_logging_config(tmp_path))
    agent.adopt_saved_session(sid, d)
    assert agent.restore_saved_stats() is None
    assert agent.stats.context_tokens == 0


def test_overflow_message_only_past_detected_ceiling() -> None:
    agent = build_test_agent_loop()
    agent.stats.context_tokens = 410_000
    assert agent.context_overflow_message() is None  # ceiling unknown: not our call
    alias = agent.config.get_active_model().alias
    agent._detected_context_sizes[alias] = 350_000
    msg = agent.context_overflow_message()
    assert msg is not None
    assert "410,000" in msg and "350,000" in msg and "/rewind" in msg
    agent.stats.context_tokens = 349_999
    assert agent.context_overflow_message() is None


@pytest.mark.asyncio
async def test_compact_is_refused_when_the_conversation_cannot_be_read() -> None:
    """Auto-compact at or past the server's ceiling would send the whole
    conversation to the same server for a summary. Stop the turn instead;
    nothing goes on the wire.
    """
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=1))
    agent = build_test_agent_loop(config=cfg)
    agent.stats.context_tokens = 400_000
    agent._detected_context_sizes[agent.config.get_active_model().alias] = 350_000

    result = MiddlewareResult(action=MiddlewareAction.COMPACT)
    events = [ev async for ev in agent._handle_middleware_result(result)]

    assert result.action == MiddlewareAction.STOP
    assert len(events) == 1
    assert isinstance(events[0], AssistantEvent)
    assert events[0].stopped_by_middleware
    assert "does not fit" in (events[0].content or "")
    assert agent.backend._requests_messages == []
    assert agent.backend._count_tokens_calls == []


# ── the --context report ──────────────────────────────────────────────────


def _report(**overrides: Any) -> str:
    base: dict[str, Any] = dict(
        session_id="abc12345-0000-0000-0000-000000000000",
        name="big one",
        model_alias="local",
        context_tokens=120_000,
        max_tokens=350_000,
        max_from_config=False,
        saved_at="2026-11-06T20:02:30+00:00",
    )
    return format_context_report(**{**base, **overrides})


def test_report_fits() -> None:
    text = _report()
    assert text.startswith('Session abc12345 "big one" on local\n')
    assert "Context: 120,000 of 350,000 tokens (34%), as of 2026-11-06" in text
    assert OVERFLOW_ADVICE not in text


def test_report_overflow_says_what_to_do() -> None:
    text = _report(context_tokens=410_000)
    assert "410,000 of 350,000 tokens (117%). " + OVERFLOW_ADVICE in text
    assert "as of" not in text
    assert OVERFLOW_ADVICE in _report(context_tokens=350_000)  # at the ceiling is over


def test_report_degrades_when_numbers_are_missing() -> None:
    assert "unknown (this session never completed a turn)" in _report(context_tokens=None)
    assert "120,000 tokens (model context size unknown)" in _report(max_tokens=0)
    assert "(from config)" in _report(max_from_config=True)
    assert "as of" not in _report(saved_at="garbage")


def test_build_report_uses_saved_stats_and_configured_ceiling() -> None:
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=200_000))
    cfg.auto_detect_context_size = False  # no server in a unit test
    alias = cfg.active_model
    metadata = {
        "session_id": "abc12345-0000-0000-0000-000000000000",
        "active_model": alias,
        "stats": {"context_tokens": 50_000},
        "end_time": "2026-11-06T20:02:30+00:00",
    }
    text = build_context_report(cfg, metadata)
    assert f"on {alias}" in text
    assert "50,000 of 200,000 tokens (from config) (25%)" in text


def test_run_context_query_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = build_test_vibe_config(models=make_test_models(auto_compact_threshold=100))
    cfg.auto_detect_context_size = False
    assert run_context_query(cfg, None) == 1
    assert "--resume SESSION_ID" in capsys.readouterr().err
    meta = {"session_id": "x", "stats": {"context_tokens": 10}}
    assert run_context_query(cfg, (meta, Path("/nowhere"))) == 0
    meta["stats"]["context_tokens"] = 100
    assert run_context_query(cfg, (meta, Path("/nowhere"))) == 2
    assert OVERFLOW_ADVICE in capsys.readouterr().out


def test_parse_arguments_context_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["privibe", "--context", "--resume", "abc12345"])
    args = parse_arguments()
    assert args.context is True
    assert args.resume == "abc12345"
    monkeypatch.setattr("sys.argv", ["privibe"])
    assert parse_arguments().context is False


# ── the server's own refusal ──────────────────────────────────────────────


def _backend_error(status: int, body: str, parsed: str | None) -> BackendError:
    return BackendError(
        provider="llamacpp",
        endpoint="http://localhost:8080/v1/chat/completions",
        status=status,
        reason="Bad Request",
        headers={},
        body_text=body,
        parsed_error=parsed,
        model="m",
        payload_summary=PayloadSummary(
            model="m", message_count=1, approx_chars=5, temperature=0.0, has_tools=False, tool_choice=None
        ),
    )


def test_llamacpp_context_refusal_is_explained() -> None:
    msg = (
        "the request exceeds the available context size. try increasing the "
        "context size or enable context shift"
    )
    err = _backend_error(400, json.dumps({"error": {"message": msg}}), msg)
    text = str(err)
    assert text.startswith("The conversation does not fit the model's context window")
    assert "/rewind" in text
    assert msg in text


def test_other_400s_keep_the_full_dump() -> None:
    err = _backend_error(400, "{}", "Assistant message must contain either content or tool_calls")
    assert str(err).startswith("LLM backend error [llamacpp]")
