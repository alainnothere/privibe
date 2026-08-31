from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from privibe.core.tools.base import BaseToolState, ToolPermission
from privibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from privibe.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileState,
    ReadFileToolConfig,
)
from privibe.core.tools.manager import ToolManager
from privibe.core.tools.permissions import PermissionContext, PermissionScope
from privibe.core.tools.utils import resolve_file_tool_permission


def _escalated(result: PermissionContext):
    return [rp for rp in result.required_permissions if rp.escalated]


# ---------------------------------------------------------------------------
# resolve_file_tool_permission
# ---------------------------------------------------------------------------


class TestFileToolResolution:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.workdir = tmp_path / "repo"
        self.outside = tmp_path / "elsewhere"
        self.workdir.mkdir()
        self.outside.mkdir()
        monkeypatch.chdir(self.workdir)

    def _resolve(self, path, **kwargs):
        kwargs.setdefault("protect_outside_workdir", True)
        kwargs.setdefault("outside_workdir_exempt", [])
        return resolve_file_tool_permission(
            path,
            tool_name="read_file",
            allowlist=[],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            **kwargs,
        )

    def test_outside_path_escalates(self):
        result = self._resolve(str(self.outside / "secret.md"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ASK
        assert len(_escalated(result)) == 1
        assert _escalated(result)[0].scope is PermissionScope.OUTSIDE_DIRECTORY

    def test_inside_path_unaffected(self):
        assert self._resolve("src/main.py") is None

    def test_exempt_outside_path_asks_without_escalation(self):
        result = self._resolve(
            str(self.outside / "scratch.md"),
            outside_workdir_exempt=[str(self.outside)],
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert not _escalated(result)
        assert any(
            r.scope is PermissionScope.OUTSIDE_DIRECTORY
            for r in result.required_permissions
        )

    def test_exempt_glob_entry(self):
        result = self._resolve(
            str(self.outside / "scratch.md"),
            outside_workdir_exempt=["*elsewhere*"],
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert not _escalated(result)

    def test_flag_off_keeps_plain_ask(self):
        result = self._resolve(
            str(self.outside / "secret.md"), protect_outside_workdir=False
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert not _escalated(result)

    def test_read_file_tool_wiring(self):
        tool = ReadFile(
            config=ReadFileToolConfig(
                protect_outside_workdir=True, outside_workdir_exempt=["/dev/*"]
            ),
            state=ReadFileState(),
        )
        result = tool.resolve_permission(
            ReadFileArgs(path=str(self.outside / "secret.md"))
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBashOutsideWorkdir:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.workdir = tmp_path / "repo"
        self.outside = tmp_path / "elsewhere"
        self.workdir.mkdir()
        self.outside.mkdir()
        monkeypatch.chdir(self.workdir)

    def _bash(self, **kwargs):
        kwargs.setdefault("protect_outside_workdir", True)
        # pytest's tmp_path lives under /tmp, which the default exemptions
        # cover — drop /tmp/* so the "outside" fixture dirs actually count
        # as outside in tests that expect an escalation.
        kwargs.setdefault("outside_workdir_exempt", ["/dev/*"])
        return Bash(config=BashToolConfig(**kwargs), state=BaseToolState())

    def test_allowlisted_cat_outside_escalates(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"cat {self.outside}/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)
        assert _escalated(result)[0].scope is PermissionScope.OUTSIDE_DIRECTORY

    def test_non_path_command_outside_escalates(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"awk '{{print}}' {self.outside}/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)

    def test_dev_null_redirect_exempt_by_default(self):
        result = self._bash().resolve_permission(
            BashArgs(command="git status > /dev/null")
        )
        assert result is not None
        assert result.permission is not ToolPermission.NEVER
        assert not _escalated(result)

    def test_tmp_exempt_by_default(self):
        bash = Bash(
            config=BashToolConfig(protect_outside_workdir=True),
            state=BaseToolState(),
        )
        result = bash.resolve_permission(BashArgs(command="cat /tmp/scratch.txt"))
        assert result is not None
        assert result.permission is not ToolPermission.NEVER
        assert not _escalated(result)

    def test_tilde_escalates(self):
        result = self._bash().resolve_permission(BashArgs(command="ls ~/stuff"))
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)

    def test_parent_traversal_escalates(self):
        result = self._bash().resolve_permission(BashArgs(command="ls .."))
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)

    def test_home_var_escalates(self, monkeypatch):
        monkeypatch.setenv("HOME", str(self.outside))
        result = self._bash().resolve_permission(
            BashArgs(command="cat $HOME/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)

    def test_flag_value_form_escalates(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"diff --from-file={self.outside}/a.md b.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)

    def test_inside_workdir_commands_still_always(self):
        (self.workdir / "README.md").write_text("hi")
        result = self._bash().resolve_permission(
            BashArgs(command="cat README.md && git status")
        )
        assert result is not None
        assert result.permission is ToolPermission.ALWAYS

    def test_bare_words_are_not_paths(self):
        result = self._bash().resolve_permission(
            BashArgs(command="git commit -m update")
        )
        assert result is not None
        assert not _escalated(result)

    def test_custom_exempt_entry(self):
        bash = self._bash(outside_workdir_exempt=["/dev/*", str(self.outside)])
        result = bash.resolve_permission(
            BashArgs(command=f"cat {self.outside}/notes.md")
        )
        assert result is not None
        assert not _escalated(result)

    def test_flag_off_outside_stays_plain_ask(self):
        bash = self._bash(protect_outside_workdir=False)
        result = bash.resolve_permission(
            BashArgs(command=f"cat {self.outside}/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert not _escalated(result)

    def test_protected_paths_still_hard_deny(self):
        bash = self._bash(protected_paths=[str(self.outside)])
        result = bash.resolve_permission(
            BashArgs(command=f"cat {self.outside}/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER
        assert "protected path" in (result.reason or "")

    def test_escalates_even_with_blanket_always_permission(self):
        bash = self._bash(permission=ToolPermission.ALWAYS)
        result = bash.resolve_permission(
            BashArgs(command=f"cat {self.outside}/secret.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert _escalated(result)


# ---------------------------------------------------------------------------
# ToolManager propagation
# ---------------------------------------------------------------------------


class TestManagerPropagation:
    def test_global_flag_propagates_to_tool_configs(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
        vibe_config.protect_outside_workdir = True
        manager = ToolManager(lambda: vibe_config)

        for tool_name in ("bash", "grep", "read_file"):
            assert manager.get_tool_config(tool_name).protect_outside_workdir

    def test_global_exempt_unioned_with_tool_defaults(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
        vibe_config.protect_outside_workdir = True
        vibe_config.outside_workdir_exempt = ["/media/usb"]
        manager = ToolManager(lambda: vibe_config)

        exempt = manager.get_tool_config("bash").outside_workdir_exempt
        assert "/dev/*" in exempt
        assert "/tmp/*" in exempt
        assert "/media/usb" in exempt

    def test_flag_defaults_off(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
        manager = ToolManager(lambda: vibe_config)
        assert not manager.get_tool_config("bash").protect_outside_workdir
