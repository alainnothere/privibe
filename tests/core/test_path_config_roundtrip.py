"""Roundtrip test pinning the [paths] template to PathConfig defaults.

WHY THIS TEST EXISTS:
    PathConfig (privibe/core/config/_settings.py) is the runtime source of
    truth for path-translation defaults; default_config.toml
    (privibe/core/config/default_config.toml) is the user-facing documented
    template that ships in every new ~/.privibe/config.toml on first launch.
    Both have the same defaults written down. If you change one and forget
    the other, this test fails — that's the point. Update both.
"""

from __future__ import annotations

import tomllib

from privibe.core.config import PATHS_TEMPLATE_FILE, PathConfig


_DRIFT_HINT = (
    "PathConfig defaults and default_config.toml have drifted. "
    "Update privibe/core/config/_settings.py:PathConfig and "
    "privibe/core/config/default_config.toml together — they document the "
    "same surface and a roundtrip test pins them on purpose."
)


def test_template_paths_section_matches_pathconfig_defaults() -> None:
    with PATHS_TEMPLATE_FILE.open("rb") as f:
        template_data = tomllib.load(f)

    template_paths = template_data.get("paths", {})

    assert template_paths == PathConfig().model_dump(), _DRIFT_HINT


def test_template_aliases_subtable_is_empty_by_default() -> None:
    """Active aliases would be silently shipped to every user; the example
    entries in the template must stay commented out."""
    with PATHS_TEMPLATE_FILE.open("rb") as f:
        template_data = tomllib.load(f)

    aliases = template_data.get("paths", {}).get("aliases", {})
    assert aliases == {}, (
        "default_config.toml ships an active alias to every user's first install. "
        "Comment example entries out (prefix each with '#') unless you really "
        "intend to ship a global default mapping."
    )


def test_template_explains_why_two_sources_exist() -> None:
    """If someone removes the warning, future-us won't know to update both
    places when defaults change. Keep at least the two key signals."""
    text = PATHS_TEMPLATE_FILE.read_text(encoding="utf-8")
    assert "PathConfig" in text
    assert "test" in text.lower()


def test_bootstrap_writes_parseable_config_with_paths_section(
    tmp_path, monkeypatch
) -> None:
    """End-to-end: bootstrap_config_files produces a TOML file where the
    top-level keys remain top-level and [paths] survives intact.

    This guards against a TOML scope-leak bug where prepending the template
    (which ends with [paths.aliases]) caused every subsequent top-level key
    from tomli_w.dump to be silently nested under paths.aliases.
    """
    import tomllib

    from privibe.core.config.harness_files import (
        init_harness_files_manager,
        reset_harness_files_manager,
    )
    from privibe.core.paths import _vibe_home

    monkeypatch.setenv("VIBE_HOME", str(tmp_path))
    monkeypatch.setattr(_vibe_home, "_DEFAULT_VIBE_HOME", tmp_path)
    reset_harness_files_manager()
    init_harness_files_manager()

    from privibe.cli.cli import bootstrap_config_files

    try:
        bootstrap_config_files()
    finally:
        reset_harness_files_manager()
    cfg = tmp_path / "config.toml"
    assert cfg.exists()
    data = tomllib.loads(cfg.read_text())

    # paths must be a top-level table with the documented defaults
    assert data["paths"]["enable_translation"] is True
    assert data["paths"]["aliases"] == {}
    # other top-level keys must NOT be nested under paths.aliases
    assert "active_model" in data
    assert "active_model" not in data["paths"]["aliases"]
