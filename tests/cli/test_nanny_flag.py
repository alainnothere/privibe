from __future__ import annotations

from pathlib import Path

import pytest

from privibe.cli.cli import run_cli
from privibe.cli.entrypoint import parse_arguments
from privibe.core.agents.models import BuiltinAgentName
from tests.cli.test_model_flag import _write_router_config


def test_parse_arguments_nanny_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["privibe", "--nanny", "router"])
    assert parse_arguments().nanny == "router"

    monkeypatch.setattr("sys.argv", ["privibe"])
    assert parse_arguments().nanny is None


def test_run_cli_nanny_override_reaches_the_interactive_loop_and_is_pinned(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")
    monkeypatch.setattr("privibe.cli.cli.get_prompt_from_stdin", lambda: None)

    captured: dict = {}

    def fake_run_textual_ui(*, agent_loop, **kwargs):
        captured["loop"] = agent_loop

    monkeypatch.setattr(
        "privibe.cli.textual_ui.app.run_textual_ui", fake_run_textual_ui
    )
    monkeypatch.setattr("sys.argv", ["privibe", "--nanny", "router"])

    run_cli(parse_arguments())

    loop = captured["loop"]
    assert loop.base_config.nanny_model == "router"
    assert loop.pinned_run_state == frozenset({"nanny"})


def test_run_cli_without_nanny_flag_pins_nothing(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setattr("privibe.cli.cli.get_prompt_from_stdin", lambda: None)

    captured: dict = {}
    monkeypatch.setattr(
        "privibe.cli.textual_ui.app.run_textual_ui",
        lambda *, agent_loop, **kwargs: captured.update(loop=agent_loop),
    )
    monkeypatch.setattr("sys.argv", ["privibe"])

    run_cli(parse_arguments())

    assert captured["loop"].base_config.nanny_model is None
    assert captured["loop"].pinned_run_state == frozenset()


def test_run_cli_unknown_nanny_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setattr("sys.argv", ["privibe", "--nanny", "nope"])

    with pytest.raises(SystemExit) as excinfo:
        run_cli(parse_arguments())
    assert excinfo.value.code == 1
    assert "--nanny: model 'nope' is not configured" in capsys.readouterr().err


def test_run_cli_nanny_without_api_key_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_router_config(config_dir)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["privibe", "--nanny", "router"])

    with pytest.raises(SystemExit) as excinfo:
        run_cli(parse_arguments())
    assert excinfo.value.code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def _run_programmatic_capturing(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> dict:
    captured: dict = {}

    def fake_run_programmatic(**kwargs):
        captured.update(kwargs)
        return "done"

    monkeypatch.setattr("privibe.cli.cli.run_programmatic", fake_run_programmatic)
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        run_cli(parse_arguments())
    assert excinfo.value.code == 0
    return captured


def test_programmatic_mode_with_a_nanny_keeps_the_asking_agent(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")

    captured = _run_programmatic_capturing(
        monkeypatch, ["privibe", "-p", "hello", "--nanny", "router"]
    )

    assert captured["nanny_alias"] == "router"
    # auto-approve would leave the nanny nothing to review.
    assert captured["agent_name"] == BuiltinAgentName.DEFAULT


def test_programmatic_mode_without_a_nanny_still_auto_approves(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)

    captured = _run_programmatic_capturing(monkeypatch, ["privibe", "-p", "hello"])

    assert captured["nanny_alias"] is None
    assert captured["agent_name"] == BuiltinAgentName.AUTO_APPROVE


def test_nanny_brief_flag_overrides_the_config_brief(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")

    captured = _run_programmatic_capturing(
        monkeypatch,
        [
            "privibe", "-p", "hello", "--nanny", "router",
            "--nanny-brief", "  rm inside build/ is fine  ",
        ],
    )

    assert captured["config"].nanny_brief == "rm inside build/ is fine"


def test_nanny_brief_without_a_nanny_is_an_error_in_programmatic_mode(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setattr(
        "sys.argv", ["privibe", "-p", "hello", "--nanny-brief", "be nice"]
    )

    with pytest.raises(SystemExit) as excinfo:
        run_cli(parse_arguments())
    assert excinfo.value.code == 1
    assert "--nanny-brief needs --nanny" in capsys.readouterr().err
