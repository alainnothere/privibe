from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from privibe.core.skills.manager import SkillManager
from privibe.core.skills.models import SkillInfo
from privibe.core.skills.render import load_skill_content
from privibe.core.tools.base import BaseToolState, InvokeContext
from privibe.core.tools.builtins.skill import Skill, SkillArgs, SkillToolConfig
from tests.mock.utils import collect_result
from tests.skills.conftest import create_skill

BODY = "## Instructions\n\nDo the thing."


def _skill_info(skills_dir: Path, name: str = "my-skill") -> SkillInfo:
    skill_dir = create_skill(skills_dir, name, body=BODY)
    return SkillInfo(
        name=name, description="A test skill", skill_path=skill_dir / "SKILL.md"
    )


def test_load_strips_frontmatter(skills_dir: Path) -> None:
    info = _skill_info(skills_dir)

    content = load_skill_content(info)

    assert "Do the thing." in content
    assert "description: A test skill" not in content
    assert "---" not in content


def test_load_wraps_in_skill_content_block(skills_dir: Path) -> None:
    info = _skill_info(skills_dir)

    content = load_skill_content(info)

    assert content.startswith('<skill_content name="my-skill">')
    assert content.endswith("</skill_content>")
    assert f"Base directory for this skill: {info.skill_dir}" in content


@pytest.mark.asyncio
async def test_slash_dispatch_and_tool_render_identically(skills_dir: Path) -> None:
    """The /slash paste and the skill tool result must be byte-identical, so the
    model recognizes an already-loaded skill no matter how it was invoked.
    """
    info = _skill_info(skills_dir)
    manager = MagicMock(spec=SkillManager)
    manager.available_skills = {"my-skill": info}
    manager.get_skill.side_effect = lambda n: {"my-skill": info}.get(n)
    ctx = InvokeContext(tool_call_id="test-call", skill_manager=manager)
    tool = Skill(config=SkillToolConfig(), state=BaseToolState())

    result = await collect_result(tool.run(SkillArgs(name="my-skill"), ctx))

    assert result.content == load_skill_content(info)
