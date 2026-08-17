from __future__ import annotations

import hashlib
import json
from pathlib import Path

from privibe.core.prompts import SystemPrompt, UtilityPrompt
from privibe.core.prompts.sync import MANIFEST_FILENAME, sync_default_prompts

ALL_PROMPT_NAMES = [f"{p.value}.md" for p in [*SystemPrompt, *UtilityPrompt]]


def _prompts_dir(config_dir: Path) -> Path:
    return config_dir / "prompts"


def _manifest(config_dir: Path) -> dict[str, str]:
    return json.loads(
        (_prompts_dir(config_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def test_first_run_copies_all_prompts(config_dir: Path) -> None:
    written = sync_default_prompts()

    assert written == len(ALL_PROMPT_NAMES)
    for name, prompt in zip(
        ALL_PROMPT_NAMES, [*SystemPrompt, *UtilityPrompt], strict=True
    ):
        local = _prompts_dir(config_dir) / name
        assert local.read_bytes() == prompt.path.read_bytes()
    manifest = _manifest(config_dir)
    assert sorted(manifest) == sorted(ALL_PROMPT_NAMES)


def test_second_run_is_a_no_op(config_dir: Path) -> None:
    sync_default_prompts()
    assert sync_default_prompts() == 0


def test_untouched_copy_refreshes_when_shipped_changes(config_dir: Path) -> None:
    sync_default_prompts()

    # Simulate a pre-upgrade state: the local copy and the manifest both
    # reflect an older shipped version that differs from the current one.
    local = _prompts_dir(config_dir) / "cli.md"
    old_shipped = b"old shipped prompt"
    local.write_bytes(old_shipped)
    manifest_path = _prompts_dir(config_dir) / MANIFEST_FILENAME
    manifest = _manifest(config_dir)
    manifest["cli.md"] = hashlib.sha256(old_shipped).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    written = sync_default_prompts()

    assert written == 1
    assert local.read_bytes() == SystemPrompt.CLI.path.read_bytes()


def test_user_edited_copy_is_never_overwritten(config_dir: Path) -> None:
    sync_default_prompts()

    local = _prompts_dir(config_dir) / "cli.md"
    local.write_text("my witty custom prompt", encoding="utf-8")

    written = sync_default_prompts()

    assert written == 0
    assert local.read_text(encoding="utf-8") == "my witty custom prompt"


def test_deleted_copy_stays_deleted(config_dir: Path) -> None:
    sync_default_prompts()

    local = _prompts_dir(config_dir) / "cli.md"
    local.unlink()

    written = sync_default_prompts()

    assert written == 0
    assert not local.exists()


def test_preexisting_empty_file_is_seeded(config_dir: Path) -> None:
    # An empty file is an accident, not authorship: Prompt.read ignores empty
    # overrides, so seeding it is strictly an improvement.
    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    local = prompts_dir / "cli.md"
    local.write_text("   \n", encoding="utf-8")

    sync_default_prompts()

    assert local.read_bytes() == SystemPrompt.CLI.path.read_bytes()


def test_empty_override_warns_only_once(config_dir: Path, caplog) -> None:
    import logging

    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "cli.md").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        SystemPrompt.CLI.read()
        SystemPrompt.CLI.read()

    warnings = [r for r in caplog.records if "is empty" in r.getMessage()]
    assert len(warnings) == 1


def test_preexisting_user_file_survives_first_sync(config_dir: Path) -> None:
    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    local = prompts_dir / "cli.md"
    local.write_text("prompt from before this feature", encoding="utf-8")

    sync_default_prompts()

    assert local.read_text(encoding="utf-8") == "prompt from before this feature"


def test_corrupt_manifest_does_not_clobber_local_files(config_dir: Path) -> None:
    sync_default_prompts()
    local = _prompts_dir(config_dir) / "cli.md"
    local.write_text("user prompt", encoding="utf-8")
    (_prompts_dir(config_dir) / MANIFEST_FILENAME).write_text(
        "not json", encoding="utf-8"
    )

    sync_default_prompts()

    assert local.read_text(encoding="utf-8") == "user prompt"


# --- read-side resolution ---


def test_read_prefers_local_copy(config_dir: Path) -> None:
    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "cli.md").write_text("LOCAL PROMPT WINS", encoding="utf-8")

    assert SystemPrompt.CLI.read() == "LOCAL PROMPT WINS"


def test_read_falls_back_to_packaged_when_no_local_copy(config_dir: Path) -> None:
    packaged = SystemPrompt.CLI.path.read_text(encoding="utf-8").strip()
    assert SystemPrompt.CLI.read() == packaged


def test_read_treats_empty_local_copy_as_missing(config_dir: Path) -> None:
    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "cli.md").write_text("   \n", encoding="utf-8")

    packaged = SystemPrompt.CLI.path.read_text(encoding="utf-8").strip()
    assert SystemPrompt.CLI.read() == packaged


def test_read_applies_to_utility_prompts_too(config_dir: Path) -> None:
    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "compact.md").write_text("CUSTOM COMPACT", encoding="utf-8")

    assert UtilityPrompt.COMPACT.read() == "CUSTOM COMPACT"


def test_mode_reminder_override_keeps_substitution(config_dir: Path) -> None:
    from privibe.core.middleware import make_plan_agent_reminder

    prompts_dir = _prompts_dir(config_dir)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "plan_reminder.md").write_text(
        "CUSTOM PLAN RULES at $plan_file_path", encoding="utf-8"
    )

    message = make_plan_agent_reminder("/tmp/my-plan.md")

    assert "CUSTOM PLAN RULES at /tmp/my-plan.md" in message
