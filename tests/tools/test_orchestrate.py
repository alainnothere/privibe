from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from privibe.core.agents.manager import AgentManager
from privibe.core.agents.models import BUILTIN_AGENTS, AgentType
from privibe.core.orchestrate.prompt import HEAD_TAIL_SEPARATOR
from privibe.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.builtins.orchestrate import (
    Orchestrate,
    OrchestrateArgs,
    OrchestrateResult,
    OrchestrateToolConfig,
)
from privibe.core.tools.builtins.task import Task, TaskArgs, TaskToolConfig
from privibe.core.types import AssistantEvent, LLMMessage, Role
from tests.conftest import build_test_vibe_config
from tests.mock.utils import collect_result

PLAN = "PLAN:\nEdit t.py.\nTASKS:\n1. make f return 2\n"
BRAINSTORM = "## Core difficulty\nnone\n"


def _check_cmd(target: Path) -> str:
    code = (
        "import importlib.util as u;"
        f"s=u.spec_from_file_location('t',r'{target}');m=u.module_from_spec(s);"
        "s.loader.exec_module(m);raise SystemExit(0 if m.f()==2 else 1)"
    )
    return f'"{sys.executable}" -c "{code}"'


class ScriptedLoops:
    """Stands in for the AgentLoop class: each construction is one role call.
    Picks the answer from the role header at the end of the message, records
    every message, effort stamp and answer_only flag, and edits the target
    file when it plays the worker.
    """

    def __init__(
        self, answers: dict[str, list[str]], target: Path | None = None
    ) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.target = target
        self.messages_seen: list[str] = []
        self.efforts: list[str] = []
        self.answer_only_flags: list[bool] = []
        self.configs: list[object] = []

    def __call__(self, config, agent_name, entrypoint_metadata=None):
        self.configs.append(config)
        loop = MagicMock()
        loop.set_approval_callback = MagicMock()
        loop.add_session_rule = MagicMock()
        loop.set_reasoning_effort = lambda e: self.efforts.append(e)
        loop.messages = []

        async def act(message: str, *, answer_only: bool = False):
            self.messages_seen.append(message)
            self.answer_only_flags.append(answer_only)
            tail = message.rpartition(HEAD_TAIL_SEPARATOR)[2]
            if "# Your role: manager, planning" in tail:
                role = "plan"
            elif "# Your role: brainstorm" in tail:
                role = "brainstorm"
            elif "# Your role: manager, curating" in tail:
                role = "manage"
            else:
                role = "execute"
                if self.target is not None:
                    self.target.write_text("def f():\n    return 2\n")
            answer = self.answers[role].pop(0)
            loop.messages = [
                LLMMessage(role=Role.system, content="s"),
                LLMMessage(role=Role.user, content=message),
                LLMMessage(role=Role.assistant, content=answer),
            ]
            yield AssistantEvent(content=answer)

        loop.act = act
        return loop


@pytest.fixture
def tool() -> Orchestrate:
    return Orchestrate(config=OrchestrateToolConfig(), state=BaseToolState())


@pytest.fixture
def ctx(tmp_path: Path) -> InvokeContext:
    config = build_test_vibe_config(
        include_project_context=False, include_prompt_detail=False
    )
    return InvokeContext(
        tool_call_id="call-1",
        agent_manager=AgentManager(lambda: config),
        session_dir=tmp_path / "session",
    )


class TestPermissions:
    def test_protected_file_is_never(self) -> None:
        t = Orchestrate(
            config=OrchestrateToolConfig(protected_paths=["/etc"]),
            state=BaseToolState(),
        )
        res = t.resolve_permission(OrchestrateArgs(goal="g", files=["/etc/passwd"]))
        assert res is not None and res.permission is ToolPermission.NEVER

    def test_done_command_touching_protected_path_is_never(self) -> None:
        t = Orchestrate(
            config=OrchestrateToolConfig(protected_paths=["~/.ssh"]),
            state=BaseToolState(),
        )
        res = t.resolve_permission(
            OrchestrateArgs(goal="g", done_command="cat ~/.ssh/id_rsa")
        )
        assert res is not None and res.permission is ToolPermission.NEVER
        assert "done_command denied" in (res.reason or "")

    def test_outside_workdir_scope_is_never_when_protected(self) -> None:
        # Not under /tmp: that is exempt by default.
        t = Orchestrate(
            config=OrchestrateToolConfig(protect_outside_workdir=True),
            state=BaseToolState(),
        )
        res = t.resolve_permission(OrchestrateArgs(goal="g", scope=["/opt/elsewhere"]))
        assert res is not None and res.permission is ToolPermission.NEVER

    def test_plain_call_falls_through_to_config_ask(self, tool: Orchestrate) -> None:
        assert (
            tool.resolve_permission(OrchestrateArgs(goal="g", files=["x.py"])) is None
        )


class TestValidation:
    @pytest.mark.asyncio
    async def test_requires_agent_manager(self, tool: Orchestrate) -> None:
        with pytest.raises(ToolError, match="agent_manager"):
            await collect_result(
                tool.run(OrchestrateArgs(goal="g"), InvokeContext(tool_call_id="x"))
            )

    @pytest.mark.asyncio
    async def test_missing_file(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path
    ) -> None:
        with pytest.raises(ToolError, match="not a file"):
            await collect_result(
                tool.run(
                    OrchestrateArgs(goal="g", files=[str(tmp_path / "nope.py")]), ctx
                )
            )

    @pytest.mark.asyncio
    async def test_scope_must_be_directory(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path
    ) -> None:
        f = tmp_path / "a.py"
        f.write_text("x\n")
        with pytest.raises(ToolError, match="not a directory"):
            await collect_result(
                tool.run(OrchestrateArgs(goal="g", scope=[str(f)]), ctx)
            )

    @pytest.mark.asyncio
    async def test_bundle_over_cap_fails_with_measured_size(
        self, tool: Orchestrate, tmp_path: Path
    ) -> None:
        f = tmp_path / "big.py"
        f.write_text("x = 1\n" * 2000)
        config = build_test_vibe_config(
            include_project_context=False,
            include_prompt_detail=False,
            orchestrate={"bundle_bytes": 1000},
        )
        ctx = InvokeContext(
            tool_call_id="c", agent_manager=AgentManager(lambda: config)
        )
        with pytest.raises(ToolError, match=r"render to \d+ bytes, over"):
            await collect_result(
                tool.run(OrchestrateArgs(goal="g", files=[str(f)]), ctx)
            )


class TestRun:
    @pytest.mark.asyncio
    async def test_end_to_end_with_scripted_loops(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "t.py"
        target.write_text("def f():\n    return 1\n")
        loops = ScriptedLoops(
            {
                "plan": [PLAN],
                "brainstorm": [BRAINSTORM],
                "manage": [
                    "1. [todo] make f return 2\nNEXT: 1",
                    "1. [done] make f return 2\nNEXT: DONE",
                ],
                "execute": ["edited\n## Notes\nf returns 2\n## Next steps\nnone"],
            },
            target=target,
        )
        args = OrchestrateArgs(
            goal="make f return 2",
            files=[str(target)],
            scope=[str(tmp_path)],
            done_command=_check_cmd(target),
        )
        with patch("privibe.core.tools.builtins.orchestrate.AgentLoop", loops):
            events = []
            async for item in tool.run(args, ctx):
                events.append(item)
        result = events[-1]
        assert isinstance(result, OrchestrateResult)
        assert result.status == "done"
        assert result.rounds == 2 and result.calls == 5
        assert result.verifier_exit_code == 0
        assert result.changed_files == [str(target)]
        ledger = Path(result.ledger_dir)
        assert ledger == ctx.session_dir / "orchestrate" / "call-1"
        assert (ledger / "notes.md").read_text().strip() == "f returns 2"
        assert (ledger / "calls" / "003-manage.md").is_file()
        # Progress reached the UI stream.
        stream = [e.message for e in events[:-1]]
        assert "manager: planning" in stream
        assert any("worker on task 1" in m for m in stream)
        # Role calls: efforts and answer-only flags.
        assert loops.efforts == ["low", "medium", "low", "medium", "low"]
        assert loops.answer_only_flags == [True, True, True, False, True]
        # Every role call got the same frozen head.
        heads = {m.partition(HEAD_TAIL_SEPARATOR)[0] for m in loops.messages_seen}
        assert len(heads) == 1
        # Worker config: scope as allowlist on the file tools, budget applied.
        wcfg = loops.configs[0]
        assert wcfg.tools["write_file"]["permission"] == "never"
        assert wcfg.tools["write_file"]["allowlist"] == [str(tmp_path.resolve() / "*")]
        assert "bash" not in {
            k
            for k, v in wcfg.tools.items()
            if v.get("allowlist") == [str(tmp_path.resolve() / "*")]
        }
        assert wcfg.max_llm_calls_per_turn == 12

    @pytest.mark.asyncio
    async def test_unverified_without_done_command(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "t.py"
        target.write_text("x\n")
        loops = ScriptedLoops({
            "plan": [PLAN],
            "brainstorm": [BRAINSTORM],
            "manage": ["1. [done] x\nNEXT: DONE"],
        })
        with patch("privibe.core.tools.builtins.orchestrate.AgentLoop", loops):
            result = await collect_result(
                tool.run(OrchestrateArgs(goal="g", files=[str(target)]), ctx)
            )
        assert result.status == "done_unverified"
        assert result.verifier_exit_code is None
        display = Orchestrate.format_result_display(result)
        assert display.success and "done_unverified" in display.message


class TestInternalProfile:
    def test_worker_profile_is_internal_subagent(self) -> None:
        p = BUILTIN_AGENTS["orchestrate-worker"]
        assert p.agent_type is AgentType.SUBAGENT and p.internal

    def test_task_tool_refuses_internal_profile(self) -> None:
        config = build_test_vibe_config(
            include_project_context=False, include_prompt_detail=False
        )
        manager = AgentManager(lambda: config)
        assert [a.name for a in manager.get_subagents()] == ["explore"]
        task = Task(config=TaskToolConfig(), state=BaseToolState())
        ctx = InvokeContext(tool_call_id="t", agent_manager=manager)

        async def go():
            await collect_result(
                task.run(TaskArgs(task="x", agent="orchestrate-worker"), ctx)
            )

        import asyncio

        with pytest.raises(ToolError, match="Only subagents"):
            asyncio.run(go())


@pytest.mark.parametrize("enabled", [True, False])
def test_enabled_switch_controls_tools_array(enabled: bool) -> None:
    from privibe.core.config import OrchestrateConfig
    from privibe.core.tools.manager import ToolManager

    config = build_test_vibe_config(orchestrate=OrchestrateConfig(enabled=enabled))
    available = ToolManager(lambda: config).available_tools
    assert ("orchestrate" in available) is enabled
    assert "bash" in available
