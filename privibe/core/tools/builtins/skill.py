from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar

from pydantic import BaseModel, Field

from privibe.core.skills.parser import SkillParseError
from privibe.core.skills.render import load_skill_content
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.types import ToolResultEvent, ToolStreamEvent


class SkillArgs(BaseModel):
    name: str = Field(description="The name of the skill to load from available_skills")


class SkillResult(BaseModel):
    name: str = Field(description="The name of the loaded skill")
    content: str = Field(description="The full skill content block")
    skill_dir: str = Field(description="Absolute path to the skill directory")


class SkillToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class Skill(
    BaseTool[SkillArgs, SkillResult, SkillToolConfig, BaseToolState],
    ToolUIData[SkillArgs, SkillResult],
):
    description: ClassVar[str] = (
        "Load a specialized skill that provides domain-specific instructions and workflows. "
        "When you recognize that a task matches one of the available skills listed in your system prompt, "
        "use this tool to load the full skill instructions. "
        "The skill will inject detailed instructions, workflows, and access to bundled resources "
        "(scripts, references, templates) into the conversation context. "
        "If a <skill_content> block for a skill is already present in the conversation "
        "(e.g. the user invoked it as a slash command), that skill is already loaded — "
        "follow it directly instead of calling this tool again."
    )

    @classmethod
    def format_call_display(cls, args: SkillArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"Loading skill: {args.name}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error:
            return ToolResultDisplay(success=False, message=event.error)
        if not isinstance(event.result, SkillResult):
            return ToolResultDisplay(success=True, message="Skill loaded")
        return ToolResultDisplay(
            success=True, message=f"Loaded skill: {event.result.name}"
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Loading skill"

    def resolve_permission(self, args: SkillArgs) -> PermissionContext | None:
        return PermissionContext(
            permission=self.config.permission,
            required_permissions=[
                RequiredPermission(
                    scope=PermissionScope.FILE_PATTERN,
                    invocation_pattern=args.name,
                    session_pattern=args.name,
                    label=f"Load skill: {args.name}",
                )
            ],
        )

    async def run(
        self, args: SkillArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | SkillResult, None]:
        if ctx is None or ctx.skill_manager is None:
            raise ToolError("Skill manager not available")

        skill_manager = ctx.skill_manager
        skill_info = skill_manager.get_skill(args.name)

        if skill_info is None:
            available = ", ".join(sorted(skill_manager.available_skills.keys()))
            raise ToolError(
                f'Skill "{args.name}" not found. Available skills: {available or "none"}'
            )

        try:
            output = load_skill_content(skill_info)
        except (OSError, SkillParseError) as e:
            raise ToolError(f"Cannot load skill file: {e}") from e

        yield SkillResult(
            name=args.name, content=output, skill_dir=str(skill_info.skill_dir)
        )
