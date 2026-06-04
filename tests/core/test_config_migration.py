"""Tests for the on-startup config migration in privibe/core/config/migration.py.

These pin the invariants that matter:
  - Top-level keys are inserted BEFORE the first [section] header so that
    uncommenting them lands them at TOML's top-level scope.
  - Section blocks are appended at the end with their own [section] header.
  - The state file prevents re-offering the same key on a subsequent launch
    (so deleting a stub means "no thanks", not "ask again next time").
  - Migration is a no-op when the user's config already covers everything.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from privibe.core.config import migration as mig


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect MIGRATION_STATE_FILE and the pending-message global so each
    test runs in isolation."""
    monkeypatch.setattr(mig, "MIGRATION_STATE_FILE", tmp_path / ".state.json")
    monkeypatch.setattr(mig, "_pending_message", None)


def _stub_defaults(monkeypatch: pytest.MonkeyPatch, defaults: dict) -> None:
    """Replace the schema-derived defaults with a controlled fixture."""
    monkeypatch.setattr(mig, "_current_default_keys", lambda: mig._flatten(defaults))


def test_top_level_key_inserted_before_first_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(monkeypatch, {"new_top_key": True, "active_model": "x"})
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'active_model = "x"\n'
        "\n"
        "[tools.bash]\n"
        'permission = "ask"\n',
        encoding="utf-8",
    )

    new_keys = mig.run_migration_if_needed(config_path)

    assert new_keys == ["new_top_key"]
    text = config_path.read_text(encoding="utf-8")
    # The commented stub must appear BEFORE the first [section] line.
    stub_idx = text.index("# new_top_key = true")
    section_idx = text.index("[tools.bash]")
    assert stub_idx < section_idx, (
        "top-level stub must precede the first [section], "
        "otherwise uncommenting nests it inside that section"
    )


def test_section_key_appended_as_full_section_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(
        monkeypatch,
        {
            "active_model": "x",
            "tools": {"new_tool": {"permission": "always", "timeout": 30}},
        },
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'active_model = "x"\n[tools.bash]\npermission = "ask"\n',
        encoding="utf-8",
    )

    new_keys = mig.run_migration_if_needed(config_path)

    assert set(new_keys) == {"tools.new_tool.permission", "tools.new_tool.timeout"}
    text = config_path.read_text(encoding="utf-8")
    # The new section block must include its own [section] header AND come
    # AFTER the user's existing [tools.bash] section.
    assert "# [tools.new_tool]" in text
    assert text.index("[tools.bash]") < text.index("# [tools.new_tool]")


def test_offered_keys_persisted_so_second_run_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(monkeypatch, {"active_model": "x", "new_key": False})
    config_path = tmp_path / "config.toml"
    config_path.write_text('active_model = "x"\n', encoding="utf-8")

    first = mig.run_migration_if_needed(config_path)
    assert first == ["new_key"]

    # User deletes the stub — they don't want it. The next run must NOT re-add.
    config_path.write_text('active_model = "x"\n', encoding="utf-8")
    second = mig.run_migration_if_needed(config_path)
    assert second == [], "offered_keys state must suppress re-offering"


def test_noop_when_config_already_has_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(monkeypatch, {"active_model": "x", "auto_approve": True})
    config_path = tmp_path / "config.toml"
    original = 'active_model = "x"\nauto_approve = true\n'
    config_path.write_text(original, encoding="utf-8")

    new_keys = mig.run_migration_if_needed(config_path)

    assert new_keys == []
    assert config_path.read_text(encoding="utf-8") == original


def test_existing_section_gets_merge_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(
        monkeypatch,
        {"tools": {"bash": {"permission": "ask", "newly_added_key": "value"}}},
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[tools.bash]\npermission = "ask"\n', encoding="utf-8"
    )

    new_keys = mig.run_migration_if_needed(config_path)

    assert new_keys == ["tools.bash.newly_added_key"]
    text = config_path.read_text(encoding="utf-8")
    # The appended block must flag (MERGE) so the user knows uncommenting the
    # whole block would produce a duplicate-table TOML error.
    assert "(MERGE: section already exists above)" in text


def test_appended_top_level_block_uncomments_to_valid_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the splice-before-first-section rule is to ensure
    that uncommenting a top-level stub lands the key at the top level.
    Verify by stripping a leading '# ' and re-parsing."""
    _stub_defaults(
        monkeypatch,
        {"new_bool": True, "active_model": "x"},
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'active_model = "x"\n[tools.bash]\npermission = "ask"\n',
        encoding="utf-8",
    )
    mig.run_migration_if_needed(config_path)

    text = config_path.read_text(encoding="utf-8")
    # Simulate the user uncommenting the new_bool line.
    uncommented = text.replace("# new_bool = true", "new_bool = true")
    parsed = tomllib.loads(uncommented)
    assert parsed["new_bool"] is True
    # And the key MUST land at the top level, not nested under a section.
    assert "tools" in parsed and "bash" in parsed["tools"]
    assert "new_bool" not in parsed["tools"]["bash"]


def test_pending_message_set_on_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_defaults(monkeypatch, {"new_key": False, "active_model": "x"})
    config_path = tmp_path / "config.toml"
    config_path.write_text('active_model = "x"\n', encoding="utf-8")

    mig.run_migration_if_needed(config_path)

    msg = mig.pop_pending_message()
    assert msg is not None
    assert "new_key" in msg
    # pop is consume-once.
    assert mig.pop_pending_message() is None
