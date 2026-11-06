from __future__ import annotations

from pathlib import Path

import pytest

from privibe.cli.cli import programmatic_resume_flags, run_cli
from privibe.cli.entrypoint import parse_arguments
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.programmatic import ProgrammaticRunTerminated
from tests.cli.test_model_flag import _write_router_config


def parse(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr("sys.argv", ["privibe", *argv])
    return parse_arguments()


@pytest.fixture
def programmatic_calls(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, tmp_path: Path
) -> dict:
    """run_cli with the model run stubbed out; `--resume abc` finds a session."""
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")
    monkeypatch.setattr("privibe.cli.cli.get_prompt_from_stdin", lambda: None)
    monkeypatch.setattr(
        "privibe.cli.cli.load_session",
        lambda args, config: (
            ({"session_id": "abc", "nanny_model": calls["saved_nanny"]}, tmp_path)
            if args.resume
            else None
        ),
    )
    calls: dict = {"outcome": "done", "saved_nanny": None}

    def fake_run_programmatic(**kwargs):
        calls.update(kwargs)
        if isinstance(calls["outcome"], BaseException):
            raise calls["outcome"]
        return calls["outcome"]

    monkeypatch.setattr("privibe.cli.cli.run_programmatic", fake_run_programmatic)
    return calls


def exit_code_of(args) -> int:
    with pytest.raises(SystemExit) as excinfo:
        run_cli(args)
    return excinfo.value.code


def test_a_bare_resume_tells_the_model_to_continue(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict, tmp_path: Path
) -> None:
    assert exit_code_of(parse(monkeypatch, "--resume", "abc", "-p")) == 0

    assert programmatic_calls["prompt"] == "continue"
    assert programmatic_calls["session_path"] == tmp_path


def test_a_resume_with_a_prompt_sends_that_prompt(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    exit_code_of(parse(monkeypatch, "--resume", "abc", "-p", "now do the other half"))

    assert programmatic_calls["prompt"] == "now do the other half"


def test_a_prompt_that_landed_after_the_session_id_is_still_the_prompt(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    # argparse gives -p no value here; the text is the positional PROMPT.
    exit_code_of(parse(monkeypatch, "-p", "--resume", "abc", "now do the other half"))

    assert programmatic_calls["prompt"] == "now do the other half"


def test_no_prompt_and_no_session_is_still_an_error(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict, capsys
) -> None:
    assert exit_code_of(parse(monkeypatch, "-p")) == 1

    assert "No prompt provided" in capsys.readouterr().err
    assert "prompt" not in programmatic_calls


def test_the_run_is_told_which_flags_a_resume_needs_again(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    exit_code_of(
        parse(
            monkeypatch, "-p", "hi", "--nanny", "router", "--nanny-brief", " be kind ",
            "--model", "router",
        )
    )

    assert programmatic_calls["resume_flags"] == [
        "--nanny", "router", "--nanny-brief", "be kind",
    ]
    assert programmatic_calls["pinned"] == frozenset({"model"})


def test_a_plain_run_needs_no_flags_again_and_pins_nothing(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    exit_code_of(parse(monkeypatch, "-p", "hi"))

    assert programmatic_calls["resume_flags"] == []
    assert programmatic_calls["pinned"] == frozenset()


def test_an_explicit_agent_is_a_flag_the_resume_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert programmatic_resume_flags(parse(monkeypatch, "-p", "hi", "--agent", "plan")) == [
        "--agent", "plan",
    ]


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    [(KeyboardInterrupt(), 130), (ProgrammaticRunTerminated(), 143)],
)
def test_an_interrupted_run_does_not_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    programmatic_calls: dict,
    capsys,
    outcome: BaseException,
    expected_code: int,
) -> None:
    programmatic_calls["outcome"] = outcome

    assert exit_code_of(parse(monkeypatch, "-p", "hi")) == expected_code
    assert "Bye" not in capsys.readouterr().out


def test_a_resumed_session_keeps_the_nanny_it_was_saved_with(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    programmatic_calls["saved_nanny"] = "router"

    assert exit_code_of(parse(monkeypatch, "--resume", "abc", "-p")) == 0

    assert programmatic_calls["nanny_alias"] == "router"
    # Reviewed means asking: auto-approve would leave the nanny idle.
    assert programmatic_calls["agent_name"] == BuiltinAgentName.DEFAULT


def test_a_nanny_given_on_the_command_line_beats_the_saved_one(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    programmatic_calls["saved_nanny"] = "gone-for-good"

    assert exit_code_of(parse(monkeypatch, "--resume", "abc", "-p", "--nanny", "router")) == 0

    assert programmatic_calls["nanny_alias"] == "router"


def test_a_resumed_session_without_a_saved_nanny_still_auto_approves(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    exit_code_of(parse(monkeypatch, "--resume", "abc", "-p"))

    assert programmatic_calls["nanny_alias"] is None
    assert programmatic_calls["agent_name"] == BuiltinAgentName.AUTO_APPROVE


def test_a_saved_nanny_that_cannot_run_stops_the_resume(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict, capsys
) -> None:
    programmatic_calls["saved_nanny"] = "gone-for-good"

    assert exit_code_of(parse(monkeypatch, "--resume", "abc", "-p")) == 1

    assert "reviewed by the nanny 'gone-for-good'" in capsys.readouterr().err
    assert "prompt" not in programmatic_calls


def test_an_explicit_agent_is_pinned_over_the_saved_one(
    monkeypatch: pytest.MonkeyPatch, programmatic_calls: dict
) -> None:
    exit_code_of(parse(monkeypatch, "--resume", "abc", "-p", "--agent", "plan"))

    assert programmatic_calls["pinned"] == frozenset({"agent"})
