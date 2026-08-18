from __future__ import annotations

from pathlib import Path

from privibe.core.skills.models import SkillInfo
from privibe.core.skills.parser import parse_frontmatter
from privibe.core.utils.io import read_safe

_MAX_LISTED_FILES = 10


def _sample_skill_files(skill_dir: Path) -> list[str]:
    files: list[str] = []
    try:
        for entry in sorted(skill_dir.rglob("*")):
            if not entry.is_file():
                continue
            if entry.name == "SKILL.md":
                continue
            files.append(str(entry.relative_to(skill_dir)))
            if len(files) >= _MAX_LISTED_FILES:
                break
    except OSError:
        pass
    return files


def format_skill_content(name: str, body: str, skill_dir: Path) -> str:
    file_lines = "\n".join(f"<file>{f}</file>" for f in _sample_skill_files(skill_dir))
    return "\n".join([
        f'<skill_content name="{name}">',
        f"# Skill: {name}",
        "",
        body.strip(),
        "",
        f"Base directory for this skill: {skill_dir}",
        "Relative paths in this skill are relative to this base directory.",
        "Note: file list is sampled.",
        "",
        "<skill_files>",
        file_lines,
        "</skill_files>",
        "</skill_content>",
    ])


def load_skill_content(skill_info: SkillInfo) -> str:
    """Read a skill from disk and render the canonical <skill_content> block.

    Every path that injects a skill into the conversation (the skill tool and
    the /slash dispatch in each UI) must go through this, so the model sees one
    format regardless of how the skill was invoked.

    Raises OSError or SkillParseError on unreadable/malformed skill files.
    """
    raw = read_safe(skill_info.skill_path)
    _, body = parse_frontmatter(raw)
    return format_skill_content(skill_info.name, body, skill_info.skill_dir)
