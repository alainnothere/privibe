from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from privibe.cli.cli import run_cli, validate_model_override
from privibe.cli.entrypoint import parse_arguments
from privibe.core.config import ModelConfig
from tests.conftest import build_test_vibe_config, get_base_config


def test_parse_arguments_model_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["privibe", "-p", "hi", "--model", "router"]
    )
    args = parse_arguments()
    assert args.model == "router"
    assert args.prompt == "hi"

    monkeypatch.setattr("sys.argv", ["privibe", "-p", "hi"])
    args = parse_arguments()
    assert args.model is None


def test_validate_model_override_accepts_active_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")
    base = get_base_config()
    providers = [
        *base["providers"],
        {
            "name": "openrouter",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key_env_var": "OPENROUTER_API_KEY",
            "backend": "generic",
        },
    ]
    models = [
        *(ModelConfig(**m) for m in base["models"]),
        ModelConfig(name="qwen/qwen3-coder", provider="openrouter", alias="router"),
    ]
    config = build_test_vibe_config(
        providers=providers,
        models=models,
        active_model="router",
    )
    assert config.active_model == "router"
    assert validate_model_override(config, "router") is None


def test_validate_model_override_unknown_alias() -> None:
    config = build_test_vibe_config()
    error = validate_model_override(config, "bogus-alias")
    assert error is not None
    assert "bogus-alias" in error
    for model in config.models:
        assert model.alias in error


def test_validate_model_override_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    base = get_base_config()
    providers = [
        *base["providers"],
        {
            "name": "openrouter",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key_env_var": "OPENROUTER_API_KEY",
            "backend": "generic",
        },
    ]
    models = [
        *(ModelConfig(**m) for m in base["models"]),
        ModelConfig(name="qwen/qwen3-coder", provider="openrouter", alias="router"),
    ]
    config = build_test_vibe_config(
        providers=providers,
        models=models,
        active_model="router",
    )
    # Construction silently falls back to the usable mistral model.
    assert config.active_model != "router"
    error = validate_model_override(config, "router")
    assert error is not None
    assert "OPENROUTER_API_KEY" in error


def _write_router_config(config_dir: Path) -> None:
    base = get_base_config()
    base["providers"].append(
        {
            "name": "openrouter",
            "api_base": "https://openrouter.ai/api/v1",
            "api_key_env_var": "OPENROUTER_API_KEY",
            "backend": "generic",
        }
    )
    base["models"].append(
        {
            "name": "qwen/qwen3-coder",
            "provider": "openrouter",
            "alias": "router",
        }
    )
    config_file = config_dir / "config.toml"
    config_file.write_text(tomli_w.dumps(base), encoding="utf-8")


def test_run_cli_model_override_reaches_programmatic_mode(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")

    captured: dict = {}

    def fake_run_programmatic(*, config, **kwargs):
        captured["config"] = config
        return "done"

    monkeypatch.setattr(
        "privibe.cli.cli.run_programmatic", fake_run_programmatic
    )

    monkeypatch.setattr(
        "sys.argv", ["privibe", "-p", "hello", "--model", "router"]
    )
    args = parse_arguments()

    with pytest.raises(SystemExit) as excinfo:
        run_cli(args)
    assert excinfo.value.code == 0
    assert captured["config"].active_model == "router"


def test_run_cli_unknown_model_exits_with_error(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    _write_router_config(config_dir)
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock")

    monkeypatch.setattr(
        "sys.argv", ["privibe", "-p", "hello", "--model", "nope"]
    )
    args = parse_arguments()

    with pytest.raises(SystemExit) as excinfo:
        run_cli(args)
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Unknown model alias 'nope'" in captured.err
