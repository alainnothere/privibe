from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from privibe.cli.textual_ui.session_exit import (
    print_session_resume_message,
    resumable_session_id,
)
from privibe.core.types import AgentStats

SESSION_ID = "12345678-1234-1234-1234-123456789abc"


def make_logger(
    save_dir: Path, session_id: str | None = SESSION_ID, enabled: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        session_id=session_id,
        session_config=SimpleNamespace(
            save_dir=str(save_dir), session_prefix="session"
        ),
    )


def save_session(save_dir: Path, session_id: str = SESSION_ID) -> None:
    session_dir = save_dir / f"session_20261002_120000_{session_id[:8]}"
    session_dir.mkdir(parents=True)
    (session_dir / "messages.jsonl").write_text("{}\n")


def test_print_session_resume_message_skips_output_without_session_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_session_resume_message(None, AgentStats())

    assert capsys.readouterr().out == ""


def test_print_session_resume_message_prints_resume_commands_and_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_session_resume_message(
        "12345678-1234-1234-1234-123456789abc",
        AgentStats(session_prompt_tokens=14_867, session_completion_tokens=6),
    )

    assert capsys.readouterr().out == (
        "\n"
        "Total tokens used this session: input=14,867 output=6 (total=14,873)\n"
        "\n"
        "To continue this session, run: privibe --continue\n"
        "Or: privibe --resume 12345678-1234-1234-1234-123456789abc\n"
    )


def test_print_session_resume_message_prints_zero_usage_for_resumed_run_without_llm_activity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_session_resume_message("12345678", AgentStats())

    assert capsys.readouterr().out == (
        "\n"
        "Total tokens used this session: input=0 output=0 (total=0)\n"
        "\n"
        "To continue this session, run: privibe --continue\n"
        "Or: privibe --resume 12345678\n"
    )


def test_print_session_resume_message_plain_has_no_markup(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a color terminal: plain must still print no escape codes.
    monkeypatch.setenv("FORCE_COLOR", "1")
    print_session_resume_message("12345678", AgentStats(), plain=True)

    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert out.endswith(
        "To continue this session, run: privibe --continue\n"
        "Or: privibe --resume 12345678\n"
    )


def test_resumable_session_id_returns_short_id_for_saved_session(
    tmp_path: Path,
) -> None:
    save_session(tmp_path)
    assert resumable_session_id(make_logger(tmp_path)) == "12345678"


def test_resumable_session_id_none_without_messages_on_disk(tmp_path: Path) -> None:
    # Opened and closed without typing: nothing to resume.
    assert resumable_session_id(make_logger(tmp_path)) is None


def test_resumable_session_id_none_when_logging_disabled(tmp_path: Path) -> None:
    save_session(tmp_path)
    assert resumable_session_id(make_logger(tmp_path, enabled=False)) is None


def test_resumable_session_id_none_without_session_id(tmp_path: Path) -> None:
    assert resumable_session_id(make_logger(tmp_path, session_id=None)) is None
