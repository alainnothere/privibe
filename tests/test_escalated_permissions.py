"""Agent-loop behavior for escalated permissions and session denials.

Escalated permissions (protect_outside_workdir hits) must reach the approval
callback even under auto_approve; 'deny for session' rules must silence
repeat asks; and file tools share one permission group so an approval for
read_file covers grep but never bash.
"""

from __future__ import annotations

from pydantic import BaseModel

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from privibe.core.agent_loop import ToolExecutionResponse
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.tools.builtins.bash import BashArgs
from privibe.core.tools.builtins.grep import GrepArgs
from privibe.core.tools.builtins.read_file import ReadFileArgs
from privibe.core.tools.permissions import RequiredPermission
from privibe.core.types import ApprovalResponse


@pytest.fixture
def outside_dir(tmp_path, monkeypatch):
    workdir = tmp_path / "repo"
    outside = tmp_path / "elsewhere"
    workdir.mkdir()
    outside.mkdir()
    monkeypatch.chdir(workdir)
    return outside


def make_loop(outside_dir):
    config = build_test_vibe_config(
        system_prompt_id="tests", include_project_context=False
    )
    config.protect_outside_workdir = True
    # pytest tmp dirs live under /tmp; keep only /dev/* exempt so the
    # fixture's outside dir counts as outside.
    config.outside_workdir_exempt = []
    for name in ("read_file", "grep", "bash"):
        config.tools.setdefault(name, {})["outside_workdir_exempt"] = ["/dev/*"]
    return build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE
    )


class RecordingCallback:
    def __init__(self, mode: str, loop=None):
        self.mode = mode
        self.loop = loop
        self.calls: list[str] = []

    async def __call__(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ):
        self.calls.append(tool_name)
        match self.mode:
            case "allow_once":
                return (ApprovalResponse.YES, None)
            case "allow_session":
                self.loop.approve_always(tool_name, required_permissions)
                return (ApprovalResponse.YES, None)
            case "deny_session":
                self.loop.deny_always(tool_name, required_permissions)
                return (ApprovalResponse.NO, "denied for session")
            case _:
                return (ApprovalResponse.NO, "no")


@pytest.mark.asyncio
async def test_escalated_reaches_callback_under_auto_approve(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("allow_once")
    loop.set_approval_callback(cb)
    assert loop.auto_approve

    tool = loop.tool_manager.get("read_file")
    decision = await loop._should_execute_tool(
        tool, ReadFileArgs(path=str(outside_dir / "notes.md")), "t1"
    )

    assert cb.calls == ["read_file"]
    assert decision.verdict is ToolExecutionResponse.EXECUTE


@pytest.mark.asyncio
async def test_non_escalated_auto_approved_without_callback(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("allow_once")
    loop.set_approval_callback(cb)

    tool = loop.tool_manager.get("bash")
    decision = await loop._should_execute_tool(
        tool, BashArgs(command="foobar --frob"), "t1"
    )

    assert cb.calls == []
    assert decision.verdict is ToolExecutionResponse.EXECUTE


@pytest.mark.asyncio
async def test_deny_for_session_silences_repeat_asks(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("deny_session", loop)
    loop.set_approval_callback(cb)

    tool = loop.tool_manager.get("read_file")
    args = ReadFileArgs(path=str(outside_dir / "notes.md"))

    first = await loop._should_execute_tool(tool, args, "t1")
    second = await loop._should_execute_tool(tool, args, "t2")

    assert first.verdict is ToolExecutionResponse.SKIP
    assert second.verdict is ToolExecutionResponse.SKIP
    assert cb.calls == ["read_file"]
    assert "Denied for this session" in (second.feedback or "")


@pytest.mark.asyncio
async def test_session_approval_shared_across_file_tools(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("allow_session", loop)
    loop.set_approval_callback(cb)

    read_file = loop.tool_manager.get("read_file")
    grep = loop.tool_manager.get("grep")

    first = await loop._should_execute_tool(
        read_file, ReadFileArgs(path=str(outside_dir / "notes.md")), "t1"
    )
    second = await loop._should_execute_tool(
        grep, GrepArgs(pattern="joker", path=str(outside_dir)), "t2"
    )

    assert first.verdict is ToolExecutionResponse.EXECUTE
    assert second.verdict is ToolExecutionResponse.EXECUTE
    assert cb.calls == ["read_file"]


@pytest.mark.asyncio
async def test_session_denial_shared_across_file_tools(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("deny_session", loop)
    loop.set_approval_callback(cb)

    read_file = loop.tool_manager.get("read_file")
    grep = loop.tool_manager.get("grep")

    await loop._should_execute_tool(
        read_file, ReadFileArgs(path=str(outside_dir / "notes.md")), "t1"
    )
    second = await loop._should_execute_tool(
        grep, GrepArgs(pattern="joker", path=str(outside_dir)), "t2"
    )

    assert second.verdict is ToolExecutionResponse.SKIP
    assert cb.calls == ["read_file"]


@pytest.mark.asyncio
async def test_bash_not_covered_by_file_group_approval(outside_dir):
    loop = make_loop(outside_dir)
    cb = RecordingCallback("allow_session", loop)
    loop.set_approval_callback(cb)

    read_file = loop.tool_manager.get("read_file")
    bash = loop.tool_manager.get("bash")

    await loop._should_execute_tool(
        read_file, ReadFileArgs(path=str(outside_dir / "notes.md")), "t1"
    )
    await loop._should_execute_tool(
        bash, BashArgs(command=f"cat {outside_dir}/notes.md"), "t2"
    )

    assert cb.calls == ["read_file", "bash"]
