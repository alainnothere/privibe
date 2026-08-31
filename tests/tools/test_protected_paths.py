from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from privibe.core.tools.base import BaseToolState, ToolPermission
from privibe.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from privibe.core.tools.builtins.grep import Grep, GrepArgs, GrepToolConfig
from privibe.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileState,
    ReadFileToolConfig,
)
from privibe.core.tools.manager import ToolManager
from privibe.core.tools.permissions import PermissionContext, PermissionScope
from privibe.core.tools.utils import (
    is_protected_path,
    resolve_file_tool_permission,
)


# ---------------------------------------------------------------------------
# is_protected_path
# ---------------------------------------------------------------------------


class TestIsProtectedPath:
    def test_file_inside_protected_dir(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        assert is_protected_path(str(stash / "combos.md"), [str(stash)])

    def test_nested_file_inside_protected_dir(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        assert is_protected_path(str(stash / "a" / "b" / "c.md"), [str(stash)])

    def test_protected_dir_itself(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        assert is_protected_path(str(stash), [str(stash)])

    def test_exact_protected_file(self, tmp_path):
        secret = tmp_path / "secrets.md"
        assert is_protected_path(str(secret), [str(secret)])

    def test_sibling_with_shared_prefix_not_protected(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        other = tmp_path / "balatro-tips-public"
        assert not is_protected_path(str(other / "x.md"), [str(stash)])

    def test_unrelated_path_not_protected(self, tmp_path):
        assert not is_protected_path(
            str(tmp_path / "src" / "main.py"), [str(tmp_path / "balatro-tips")]
        )

    def test_glob_entry(self, tmp_path):
        path = tmp_path / "notes" / "balatro-secrets.md"
        assert is_protected_path(str(path), ["*balatro*"])
        assert not is_protected_path(str(tmp_path / "readme.md"), ["*balatro*"])

    def test_tilde_entry_expands_to_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert is_protected_path(str(tmp_path / "stash" / "f.md"), ["~/stash"])

    def test_relative_path_resolved_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        stash = tmp_path / "balatro-tips"
        assert is_protected_path("balatro-tips/combos.md", [str(stash)])

    def test_empty_list_protects_nothing(self, tmp_path):
        assert not is_protected_path(str(tmp_path / "f.md"), [])


# ---------------------------------------------------------------------------
# resolve_file_tool_permission
# ---------------------------------------------------------------------------


class TestFileToolResolution:
    def test_protected_path_returns_never(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        result = resolve_file_tool_permission(
            str(stash / "combos.md"),
            tool_name="read_file",
            allowlist=[],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            protected_paths=[str(stash)],
        )
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER
        assert "protected" in (result.reason or "")

    def test_protected_beats_allowlist(self, tmp_path):
        stash = tmp_path / "balatro-tips"
        target = stash / "combos.md"
        result = resolve_file_tool_permission(
            str(target),
            tool_name="read_file",
            allowlist=[str(target)],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            protected_paths=[str(stash)],
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_unprotected_path_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = resolve_file_tool_permission(
            "src/main.py",
            tool_name="read_file",
            allowlist=[],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            protected_paths=[str(tmp_path / "balatro-tips")],
        )
        assert result is None

    def test_grep_tool_denies_protected_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        stash = tmp_path / "balatro-tips"
        grep = Grep(
            config=GrepToolConfig(protected_paths=[str(stash)]),
            state=BaseToolState(),
        )
        result = grep.resolve_permission(GrepArgs(pattern="joker", path=str(stash)))
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_read_file_tool_denies_protected_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        stash = tmp_path / "balatro-tips"
        tool = ReadFile(
            config=ReadFileToolConfig(protected_paths=[str(stash)]),
            state=ReadFileState(),
        )
        result = tool.resolve_permission(ReadFileArgs(path=str(stash / "combos.md")))
        assert result is not None
        assert result.permission is ToolPermission.NEVER


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBashProtectedPaths:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self.workdir = tmp_path
        self.stash = tmp_path / "balatro-tips"

    def _bash(self, **kwargs):
        kwargs.setdefault("protected_paths", [str(self.stash)])
        return Bash(config=BashToolConfig(**kwargs), state=BaseToolState())

    def test_allowlisted_cat_on_protected_path_is_never(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"cat {self.stash}/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER
        assert "protected" in (result.reason or "")

    def test_relative_token_into_protected_dir_is_never(self):
        result = self._bash().resolve_permission(
            BashArgs(command="cat balatro-tips/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_home_var_expansion_is_never(self, monkeypatch):
        monkeypatch.setenv("HOME", str(self.workdir))
        result = self._bash().resolve_permission(
            BashArgs(command="cat $HOME/balatro-tips/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_flag_value_form_is_never(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"diff --from-file={self.stash}/combos.md other.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_protected_in_second_command_of_chain_is_never(self):
        result = self._bash().resolve_permission(
            BashArgs(command=f"echo hi && cat {self.stash}/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER

    def test_unrelated_allowlisted_command_still_always(self):
        (self.workdir / "README.md").write_text("hi")
        result = self._bash().resolve_permission(BashArgs(command="cat README.md"))
        assert result is not None
        assert result.permission is ToolPermission.ALWAYS

    def test_no_protected_paths_configured_is_noop(self):
        bash = self._bash(protected_paths=[])
        result = bash.resolve_permission(
            BashArgs(command="cat balatro-tips/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.ALWAYS

    def test_protected_beats_blanket_always_permission(self):
        bash = self._bash(permission=ToolPermission.ALWAYS)
        result = bash.resolve_permission(
            BashArgs(command=f"cat {self.stash}/combos.md")
        )
        assert result is not None
        assert result.permission is ToolPermission.NEVER


class TestBashEnvVarOutsideDirs:
    def test_home_var_path_outside_workdir_asks(self, tmp_path, monkeypatch):
        workdir = tmp_path / "repo"
        home = tmp_path / "home"
        (home / "notes").mkdir(parents=True)
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("HOME", str(home))

        bash = Bash(config=BashToolConfig(), state=BaseToolState())
        result = bash.resolve_permission(
            BashArgs(command="cat $HOME/notes/file.txt")
        )
        assert result is not None
        assert result.permission is ToolPermission.ASK
        assert any(
            r.scope is PermissionScope.OUTSIDE_DIRECTORY
            for r in result.required_permissions
        )


# ---------------------------------------------------------------------------
# ToolManager merge
# ---------------------------------------------------------------------------


class TestManagerMerge:
    def test_global_protected_paths_injected_into_tool_config(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
        vibe_config.protected_paths = ["/home/u/balatro-tips"]
        manager = ToolManager(lambda: vibe_config)

        for tool_name in ("bash", "grep", "read_file"):
            config = manager.get_tool_config(tool_name)
            assert config.protected_paths == ["/home/u/balatro-tips"]

    def test_global_and_per_tool_lists_are_unioned(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            tools={"grep": {"protected_paths": ["/home/u/diary"]}},
        )
        vibe_config.protected_paths = ["/home/u/balatro-tips"]
        manager = ToolManager(lambda: vibe_config)

        config = manager.get_tool_config("grep")
        assert config.protected_paths == [
            "/home/u/balatro-tips",
            "/home/u/diary",
        ]

    def test_no_global_list_leaves_tool_config_untouched(self):
        vibe_config = build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
        manager = ToolManager(lambda: vibe_config)
        assert manager.get_tool_config("grep").protected_paths == []
