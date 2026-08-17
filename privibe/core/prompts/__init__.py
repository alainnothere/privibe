from __future__ import annotations

from enum import StrEnum, auto
from pathlib import Path

from privibe import VIBE_ROOT
from privibe.core.utils.io import read_safe

_PROMPTS_DIR = VIBE_ROOT / "core" / "prompts"

# Empty-override warnings already emitted, so repeated reads of the same
# prompt (banner, agents, per-turn utilities) don't spam the log.
_warned_empty_overrides: set[Path] = set()


def _override_dirs() -> list[Path]:
    """Prompt override locations: project .privibe/prompts, then the user's.

    The harness manager may not be initialized yet (early startup, some
    tests); in that state only the packaged prompts exist to read.
    """
    try:
        from privibe.core.config.harness_files._harness_manager import (
            get_harness_files_manager,
        )

        mgr = get_harness_files_manager()
    except Exception:
        return []
    return mgr.project_prompts_dirs + mgr.user_prompts_dirs


class Prompt(StrEnum):
    @property
    def path(self) -> Path:
        return (_PROMPTS_DIR / self.value).with_suffix(".md")

    def read(self) -> str:
        for directory in _override_dirs():
            candidate = (directory / self.value).with_suffix(".md")
            if candidate.is_file():
                stripped = read_safe(candidate).strip()
                if stripped:
                    return stripped
                if candidate not in _warned_empty_overrides:
                    _warned_empty_overrides.add(candidate)
                    from privibe.core.logger import logger

                    logger.warning(
                        "Prompt override %s is empty; using the packaged prompt",
                        candidate,
                    )
        return read_safe(self.path).strip()


class SystemPrompt(Prompt):
    CLI = auto()
    EXPLORE = auto()
    TESTS = auto()


class UtilityPrompt(Prompt):
    AGENTS_DOC = auto()
    COMPACT = auto()
    DANGEROUS_DIRECTORY = auto()
    PROJECT_CONTEXT = auto()
    TURN_SUMMARY = auto()


__all__ = ["SystemPrompt", "UtilityPrompt"]
