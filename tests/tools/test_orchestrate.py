from __future__ import annotations

import asyncio
import json
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


class CancelAtManage(ScriptedLoops):
    """The scripted loops, except the manager call is a user cancel: what the
    loop sees when ctrl+c lands while the manager is thinking.
    """

    def __call__(self, config, agent_name, entrypoint_metadata=None):
        loop = super().__call__(config, agent_name, entrypoint_metadata)
        scripted_act = loop.act

        async def act(message: str, *, answer_only: bool = False):
            if "# Your role: manager, curating" in message:
                raise asyncio.CancelledError
            async for ev in scripted_act(message, answer_only=answer_only):
                yield ev

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
        # The verifier passes right after the worker, so the loop ends there:
        # plan, brainstorm, manage, execute.
        assert result.rounds == 1 and result.calls == 4
        assert result.detail == "verifier passes after a worker"
        assert result.verifier_exit_code == 0
        assert result.changed_files == [str(target)]
        ledger = Path(result.ledger_dir)
        assert ledger == ctx.session_dir / "orchestrate" / "call-1"
        assert (ledger / "notes.md").read_text().strip() == "f returns 2"
        assert (ledger / "calls" / "003-manage.md").is_file()
        # Progress reached the UI stream.
        stream = [e.message for e in events[:-1]]
        assert stream[0].startswith("── manager · plan ")
        assert any("worker · round 1/10 · task 1: make f return 2" in m for m in stream)
        # The stamp lands on the header of the call that just ended.
        stamped = [
            m
            for m in stream
            if m.startswith("── worker")
            and "call" not in m
            and m.rstrip("─ ").endswith("s")
        ]
        assert stamped
        # The ledger knows which session played which role call (the scripted
        # loops log nowhere, so the dirs are empty here).
        calls = json.loads((ledger / "calls.json").read_text())
        assert [c["label"] for c in calls] == [
            "manager · plan",
            "brainstorm",
            "manager · round 1/10",
            "worker · round 1/10 · task 1: make f return 2",
        ]
        assert [c["tag"] for c in calls] == [
            "plan",
            "brainstorm",
            "manager 1/10",
            "worker 1/10 · task 1",
        ]
        assert all(c["session_dir"] == "" for c in calls)
        # The trace ends on a rule that says how the run ended, with the detail.
        assert stream[-1].startswith("── end · done ")
        assert (
            stream[-1]
            .rstrip()
            .endswith(" 1 round · 4 calls\n  verifier passes after a worker")
        )
        # Role calls: efforts and answer-only flags.
        assert loops.efforts == ["low", "medium", "low", "medium"]
        assert loops.answer_only_flags == [True, True, True, False]
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
        # A bare DONE on an untouched workspace is overruled, so a worker
        # edits the file first; then the manager's DONE stands, unverified.
        loops = ScriptedLoops(
            {
                "plan": [PLAN],
                "brainstorm": [BRAINSTORM],
                "manage": [
                    "1. [todo] make f return 2\nNEXT: 1",
                    "1. [done] make f return 2\nNEXT: DONE",
                ],
                "execute": ["edited\n## Notes\nn\n"],
            },
            target=target,
        )
        with patch("privibe.core.tools.builtins.orchestrate.AgentLoop", loops):
            result = await collect_result(
                tool.run(OrchestrateArgs(goal="g", files=[str(target)]), ctx)
            )
        assert result.status == "done_unverified"
        assert result.detail == "manager declared done; no verifier"
        assert result.calls == 5
        assert result.changed_files == [str(target)]
        assert result.verifier_exit_code is None
        display = Orchestrate.format_result_display(result)
        assert display.success and "done_unverified" in display.message


class TestInternalProfile:
    @pytest.mark.asyncio
    async def test_resume_takes_the_job_from_the_ledger(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        target = tmp_path / "t.py"
        target.write_text("def f():\n    return 1\n")
        first = CancelAtManage(
            {"plan": [PLAN], "brainstorm": [BRAINSTORM]}, target=target
        )
        args = OrchestrateArgs(
            goal="make f return 2",
            files=[str(target)],
            scope=[str(tmp_path)],
            done_command=_check_cmd(target),
            max_rounds=2,
        )
        with (
            patch("privibe.core.tools.builtins.orchestrate.AgentLoop", first),
            pytest.raises(asyncio.CancelledError),
        ):
            async for _ in tool.run(args, ctx):
                pass
        ledger_dir = ctx.session_dir / "orchestrate" / "call-1"
        assert (ledger_dir / "job.json").is_file()
        assert (ledger_dir / "state" / "002.json").is_file()

        second = ScriptedLoops(
            {
                "manage": ["1. [todo] make f return 2\nNEXT: 1"],
                "execute": ["edited\n## Notes\nf returns 2\n## Next steps\nnone"],
            },
            target=target,
        )
        resume = OrchestrateArgs(
            resume_ledger=str(ledger_dir), direction="Keep the function name."
        )
        with patch("privibe.core.tools.builtins.orchestrate.AgentLoop", second):
            events = []
            async for item in tool.run(resume, ctx):
                events.append(item)
        result = events[-1]
        assert isinstance(result, OrchestrateResult)
        assert result.status == "done", result.detail
        assert result.ledger_dir == str(ledger_dir)
        assert result.checkpoint == 2
        # No plan, no brainstorm: the ledger had them.
        assert second.answer_only_flags == [True, False]
        assert all("Keep the function name." in m for m in second.messages_seen)
        # Scope came from job.json and still binds the worker's edit tools.
        wcfg = second.configs[0]
        assert wcfg.tools["write_file"]["allowlist"] == [str(tmp_path.resolve() / "*")]
        stream = [e.message for e in events[:-1]]
        assert stream[0].startswith("resumed from checkpoint 2 (brainstormed")
        assert (
            (ledger_dir / "steer.md").read_text().endswith("Keep the function name.\n")
        )

    @pytest.mark.asyncio
    async def test_resume_refuses_a_job_description_and_a_bare_dir(
        self, tool: Orchestrate, ctx: InvokeContext, tmp_path: Path
    ) -> None:
        with pytest.raises(ToolError, match="leave goal, files, scope"):
            async for _ in tool.run(
                OrchestrateArgs(resume_ledger=str(tmp_path), goal="again"), ctx
            ):
                pass
        with pytest.raises(ToolError, match="no resumable run"):
            async for _ in tool.run(OrchestrateArgs(resume_ledger=str(tmp_path)), ctx):
                pass
        with pytest.raises(ToolError, match="not a directory"):
            async for _ in tool.run(
                OrchestrateArgs(resume_ledger=str(tmp_path / "nope")), ctx
            ):
                pass

    def test_resume_permission_reads_the_ledger_job(self, tmp_path: Path) -> None:
        ledger_dir = tmp_path / "ledger"
        ledger_dir.mkdir()
        (ledger_dir / "job.json").write_text(
            json.dumps({
                "goal": "g",
                "files": [str(tmp_path / "secrets" / "k.txt")],
                "scope": [],
                "done_command": "",
                "max_rounds": 2,
                "cwd": str(tmp_path),
            })
        )
        tool = Orchestrate(
            config=OrchestrateToolConfig(protected_paths=["**/secrets/**"]),
            state=BaseToolState(),
        )
        ctx = tool.resolve_permission(OrchestrateArgs(resume_ledger=str(ledger_dir)))
        assert ctx is not None and ctx.permission == ToolPermission.NEVER
        # A directory that does not exist yet is not created by the check.
        missing = tmp_path / "missing"
        assert (
            tool.resolve_permission(OrchestrateArgs(resume_ledger=str(missing))) is None
        )
        assert not missing.exists()

    def test_resume_call_line_names_the_ledger(self) -> None:
        display = Orchestrate.format_call_display(
            OrchestrateArgs(resume_ledger="/tmp/x", checkpoint=3, direction="go")
        )
        assert display.summary.splitlines() == [
            "orchestrate (resume)",
            "  resume: /tmp/x",
            "  checkpoint: 3",
            "  direction: go",
        ]

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
