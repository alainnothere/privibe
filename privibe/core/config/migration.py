"""On-startup config migration: append commented stubs for new defaults.

WHAT THIS DOES
    Each launch, compare the user's ~/.privibe/config.toml against the current
    schema's defaults. Anything in the schema but not in the user's file gets
    appended as a *commented* TOML stub, plus added to a state file so we
    don't re-offer the same keys on every launch (in particular, if the user
    deletes a stub we treat that as "no thanks" and never re-add it).

    The user's behaviour does not change unless they explicitly uncomment a
    stub. The point is purely to make new options discoverable after upgrades.

WHY THE TWO-CHUNK INSERTION
    TOML scopes everything after a [section] header into that section until
    the next header. Most user configs end with a section like [tools.read_
    file], so a top-level key appended at the bottom would, when uncommented,
    be parsed as a nested key under that last section. To avoid that:

      - Top-level keys (no dot in their path) get inserted ABOVE the user's
        first [section] header, in a clearly delimited block.
      - Keys that live under a section (any dotted path) get appended at the
        END of the file as a fresh [section] block — each commented stub
        carries its own [section] header, so scope is self-contained.

    For sections that already exist in the user's config, we still append a
    fresh commented [section] block. Uncommenting both blocks would be a TOML
    duplicate-table error, so the appended block carries an inline note
    asking the user to merge instead.
"""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any

from privibe.core.logger import logger
from privibe.core.paths import VIBE_HOME

# Where we remember which keys have already been offered. Tracked so the user
# can delete an offered stub without us re-adding it next launch. Resolved
# lazily via VIBE_HOME.path so the VIBE_HOME env var override still works.
MIGRATION_STATE_FILE = VIBE_HOME.path / ".config_migration_state.json"

# Stash for the one-line message the UI / programmatic mode shows on startup.
# Set during run_migration_if_needed; cleared by pop_pending_message.
_pending_message: str | None = None


def pop_pending_message() -> str | None:
    """Read-and-clear the migration message. Used by the UI on_mount and by
    programmatic mode to surface "N new keys were added to your config" once."""
    global _pending_message
    msg = _pending_message
    _pending_message = None
    return msg


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted-key form: {'tools': {'bash': {'permission': 'ask'}}}
    becomes {'tools.bash.permission': 'ask'}. Lists are kept as-is."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _load_offered_keys() -> set[str]:
    try:
        data = json.loads(MIGRATION_STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("offered_keys", []))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()


def _save_offered_keys(keys: set[str]) -> None:
    try:
        MIGRATION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        MIGRATION_STATE_FILE.write_text(
            json.dumps({"offered_keys": sorted(keys)}, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Could not save config migration state: %s", e)


def _current_default_keys() -> dict[str, Any]:
    """Flat dict of {dotted_key: default_value} for everything the schema
    currently exposes, including tool-config sub-tables. Computed by dumping a
    default VibeConfig, then merging in whatever ToolManager.discover_tool_
    defaults() reports — same surface bootstrap_config_files writes for fresh
    installs.

    NOTE: we do NOT pop "paths" or "preflight_warmup" the way bootstrap does,
    because for migration purposes the user should still see those keys as
    available — bootstrap pops them to avoid duplication with the documented
    template, but that template is only written for fresh installs."""
    from privibe.core.config import VibeConfig

    config = VibeConfig.model_construct()
    config_dict = config.model_dump(mode="json")

    from privibe.core.tools.manager import ToolManager

    tool_defaults = ToolManager.discover_tool_defaults()
    if tool_defaults:
        config_dict["tools"] = tool_defaults

    return _flatten(config_dict)


def _user_config_keys(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        return _flatten(data)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError) as e:
        logger.warning(
            "Could not parse %s for migration check (%s); skipping.", config_path, e
        )
        return {}


def _toml_value_repr(v: Any) -> str:
    """Render a Python value as TOML scalar/array literal. Only handles the
    types tomllib produces from our config defaults: bool, int, float, str,
    None (skipped by caller), and list of those."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # Use double-quoted form; escape backslashes and quotes.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, list):
        if not v:
            return "[]"
        items = ", ".join(_toml_value_repr(x) for x in v)
        return f"[{items}]"
    # Dicts shouldn't reach here (flatten splits them); fall back to JSON-ish.
    return json.dumps(v)


def _split_top_level_vs_sectioned(
    missing: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Partition missing keys: bare top-level keys vs keys grouped by their
    parent [section]."""
    top_level: dict[str, Any] = {}
    sectioned: dict[str, dict[str, Any]] = {}
    for k, v in missing.items():
        if "." not in k:
            top_level[k] = v
        else:
            section, _, leaf = k.rpartition(".")
            sectioned.setdefault(section, {})[leaf] = v
    return top_level, sectioned


def _format_top_level_block(top_level: dict[str, Any], stamp: str) -> str:
    if not top_level:
        return ""
    lines = [
        "",
        "# ============================================================================",
        f"# privibe config migration ({stamp}) — new top-level options",
        "# These keys are new since you last upgraded. Uncomment to opt in.",
        "# Delete this block if you don't want them; privibe won't re-offer the same keys.",
        "# ----------------------------------------------------------------------------",
    ]
    for k in sorted(top_level):
        lines.append(f"# {k} = {_toml_value_repr(top_level[k])}")
    lines.append(
        "# ============================================================================"
    )
    lines.append("")
    return "\n".join(lines)


def _format_sectioned_block(
    sectioned: dict[str, dict[str, Any]],
    existing_sections: set[str],
    stamp: str,
) -> str:
    if not sectioned:
        return ""
    lines = [
        "",
        "# ============================================================================",
        f"# privibe config migration ({stamp}) — new section options",
        "# Each block below is a TOML [section] table that's new since your last",
        "# upgrade. Uncomment any line to opt in. Sections marked (MERGE) already",
        "# exist above in your config — uncommenting THIS block's [section] header",
        "# would be a TOML duplicate-table error; merge those keys into the existing",
        "# section instead. Sections without (MERGE) are safe to uncomment as-is.",
        "# ----------------------------------------------------------------------------",
    ]
    for section in sorted(sectioned):
        marker = "  # (MERGE: section already exists above)" if section in existing_sections else ""
        lines.append(f"# [{section}]{marker}")
        for leaf in sorted(sectioned[section]):
            v = sectioned[section][leaf]
            lines.append(f"# {leaf} = {_toml_value_repr(v)}")
        lines.append("")
    lines.append(
        "# ============================================================================"
    )
    return "\n".join(lines)


def _existing_sections(config_text: str) -> set[str]:
    """Section headers (`[a.b]` and `[[a.b]]`) currently present in the file."""
    out: set[str] = set()
    for raw in config_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[[") and line.endswith("]]"):
            out.add(line[2:-2].strip())
        elif line.startswith("[") and line.endswith("]"):
            out.add(line[1:-1].strip())
    return out


def _insert_top_level_block(content: str, block: str) -> str:
    """Splice the top-level migration block in BEFORE the user's first
    [section] / [[section]] header. If there is no section, append at end."""
    if not block:
        return content
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("[") and not s.startswith("# "):
            return "".join(lines[:i]) + block + "\n" + "".join(lines[i:])
    suffix = "" if content.endswith("\n") else "\n"
    return content + suffix + block + "\n"


def _append_sectioned_block(content: str, block: str) -> str:
    if not block:
        return content
    suffix = "" if content.endswith("\n") else "\n"
    return content + suffix + block + "\n"


def run_migration_if_needed(config_path: Path) -> list[str]:
    """Compare the user's config against current defaults; append commented
    stubs for any new keys we haven't already offered. Returns the list of
    newly-offered dotted keys (empty when nothing changed).

    Also sets the module-level _pending_message that the UI / programmatic
    mode picks up via pop_pending_message() to surface a startup banner.
    """
    if not config_path.exists():
        return []

    defaults = _current_default_keys()
    user = _user_config_keys(config_path)
    if not user:
        return []

    offered = _load_offered_keys()
    missing = {k: v for k, v in defaults.items() if k not in user and k not in offered}
    if not missing:
        return []

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read %s for migration: %s", config_path, e)
        return []

    from datetime import date

    stamp = date.today().isoformat()
    top_level, sectioned = _split_top_level_vs_sectioned(missing)
    existing = _existing_sections(content)

    new_content = _insert_top_level_block(
        content, _format_top_level_block(top_level, stamp)
    )
    new_content = _append_sectioned_block(
        new_content, _format_sectioned_block(sectioned, existing, stamp)
    )

    try:
        config_path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write migration block to %s: %s", config_path, e)
        return []

    _save_offered_keys(offered | set(missing))

    new_keys = sorted(missing)
    _set_pending_message(new_keys, config_path)
    return new_keys


def _set_pending_message(new_keys: list[str], config_path: Path) -> None:
    global _pending_message
    if not new_keys:
        return
    # Group section keys under their parent so the message stays terse for
    # configs that gained a whole sub-table at once.
    top: list[str] = []
    sections: dict[str, int] = {}
    for k in new_keys:
        if "." not in k:
            top.append(k)
        else:
            section, _, _leaf = k.rpartition(".")
            sections[section] = sections.get(section, 0) + 1
    parts: list[str] = []
    parts.extend(top)
    parts.extend(f"{s}.* ({n})" for s, n in sorted(sections.items()))
    summary = ", ".join(parts) if parts else f"{len(new_keys)} keys"
    _pending_message = (
        f"privibe config migration: {len(new_keys)} new option(s) appended to "
        f"{config_path} as commented stubs — {summary}. "
        f"Edit the file and uncomment any you want to enable."
    )
