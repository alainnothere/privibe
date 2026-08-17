"""Copy the bundled prompt files into ~/.privibe/prompts/ so they are locally
editable, and keep untouched copies fresh across upgrades.

Prompt.read() prefers the local copy (project dir, then user dir) over the
packaged file, so whatever lands here is what actually reaches the model.
A hash manifest records which shipped version each local file was copied
from, which lets upgrades refresh files the user never edited without ever
clobbering files they did:

- not in manifest, no local file -> copy shipped, record its hash
- not in manifest, local exists  -> user-authored; leave it, record hash
- in manifest, local missing     -> user deleted it; respect that
- local hash == recorded hash    -> untouched copy; refresh if shipped changed
- local hash != recorded hash    -> user-owned; never overwritten

Hashes are over raw bytes and files are copied as bytes, so line-ending
translation can never make an untouched file look user-edited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from privibe import VIBE_ROOT
from privibe.core.config.harness_files._paths import GLOBAL_PROMPTS_DIR
from privibe.core.logger import logger
from privibe.core.prompts import (
    TOOL_PROMPTS_SUBDIR,
    Prompt,
    SystemPrompt,
    UtilityPrompt,
)

MANIFEST_FILENAME = ".manifest.json"

_TOOL_PROMPTS_DIR = VIBE_ROOT / "core" / "tools" / "builtins" / "prompts"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return {str(k): str(v) for k, v in loaded.items()}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return {}


def _sync_one(
    shipped_path: Path, local: Path, key: str, manifest: dict[str, str]
) -> int:
    """Sync a single shipped prompt into its local copy. Returns files written."""
    try:
        shipped = shipped_path.read_bytes()
    except OSError as e:
        logger.warning("Cannot read bundled prompt %s: %s", shipped_path, e)
        return 0
    shipped_hash = _sha256(shipped)
    recorded = manifest.get(key)
    written = 0

    if recorded is None:
        # First sight of this prompt. A pre-existing local file is
        # user-authored and must survive; an empty one is an accident
        # (Prompt.read ignores empty overrides), so seed both cases.
        try:
            existing = local.read_bytes()
        except OSError:
            existing = b""
        if not existing.strip():
            local.write_bytes(shipped)
            written = 1
    elif local.exists():
        local_hash = _sha256(local.read_bytes())
        if local_hash == recorded and shipped_hash != recorded:
            local.write_bytes(shipped)
            written = 1
    # in manifest but missing locally: the user deleted it; respect that.

    manifest[key] = shipped_hash
    return written


def sync_default_prompts() -> int:
    """Sync bundled prompts into the user prompts dir. Returns files written."""
    prompts_dir = GLOBAL_PROMPTS_DIR.path
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prompts_dir / MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)
    written = 0

    prompts: list[Prompt] = [*SystemPrompt, *UtilityPrompt]
    for prompt in prompts:
        local = (prompts_dir / prompt.value).with_suffix(".md")
        written += _sync_one(prompt.path, local, local.name, manifest)

    # Per-tool prompts ride the same bus, namespaced under tools/ so a tool
    # file can never collide with a top-level prompt of the same name.
    tools_dir = prompts_dir / TOOL_PROMPTS_SUBDIR
    tools_dir.mkdir(parents=True, exist_ok=True)
    for shipped_path in sorted(_TOOL_PROMPTS_DIR.glob("*.md")):
        key = f"{TOOL_PROMPTS_SUBDIR}/{shipped_path.name}"
        written += _sync_one(shipped_path, tools_dir / shipped_path.name, key, manifest)

    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as e:
        logger.warning("Cannot write prompts manifest %s: %s", manifest_path, e)

    return written
