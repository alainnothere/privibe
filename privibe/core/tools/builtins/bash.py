from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from functools import lru_cache
import os
from pathlib import Path
import signal
import sys
from typing import ClassVar, Literal, final

from pydantic import BaseModel, Field
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from privibe.core.tools.arity import build_session_pattern
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
from privibe.core.tools.utils import (
    is_path_within_workdir,
    is_protected_path,
    normalize_tool_path,
)
from privibe.core.types import ToolResultEvent, ToolStreamEvent
from privibe.core.utils import (
    DEFAULT_BASH_SEARCH_PATHS,
    is_windows,
    resolve_windows_bash,
)


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def _extract_commands(command: str) -> list[str]:
    parser = _get_parser()
    tree = parser.parse(command.encode("utf-8"))

    commands: list[str] = []

    def find_commands(node: Node) -> None:
        if node.type == "command":
            parts = []
            for child in node.children:
                if (
                    child.type
                    in {"command_name", "word", "string", "raw_string", "concatenation"}
                    and child.text is not None
                ):
                    parts.append(child.text.decode("utf-8"))
            if parts:
                commands.append(" ".join(parts))

        for child in node.children:
            find_commands(child)

    find_commands(tree.root_node)
    return commands


def _get_subprocess_encoding(windows_bash: bool = False) -> str:
    if sys.platform == "win32" and not windows_bash:
        # cmd.exe output uses the OEM code page (e.g., cp850, cp1252); Git
        # Bash tools emit UTF-8.
        import ctypes

        return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
    return "utf-8"


def _get_shell_executable() -> str | None:
    if is_windows():
        return None
    return os.environ.get("SHELL")


def _get_base_env(windows_bash: bool = False) -> dict[str, str]:
    base_env = {**os.environ, "CI": "true", "NONINTERACTIVE": "1", "NO_TTY": "1"}

    if is_windows() and not windows_bash:
        base_env["GIT_PAGER"] = "more"
        base_env["PAGER"] = "more"
    else:
        base_env["TERM"] = "dumb"
        base_env["GIT_PAGER"] = "cat"
        base_env["PAGER"] = "cat"
        base_env["LESS"] = "-FX"
        if not is_windows():
            base_env["DEBIAN_FRONTEND"] = "noninteractive"
            # Not set under Git Bash: MSYS may not ship the en_US.UTF-8 locale.
            base_env["LC_ALL"] = "en_US.UTF-8"

    return base_env


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return

    try:
        if sys.platform == "win32":
            try:
                subprocess_proc = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(proc.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await subprocess_proc.wait()
            except (FileNotFoundError, OSError):
                proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

        await proc.wait()
    except (ProcessLookupError, PermissionError, OSError):
        pass


# On Windows both backends are reachable — Git Bash when found, cmd.exe as the
# fallback (and via `cmd //c` from inside bash) — so the default lists there
# are the union of both worlds. Entries for the inactive backend are harmless:
# allowlisting `cat` under cmd or denying `cmd /k` under bash never fires.


def _get_default_allowlist() -> list[str]:
    common = ["cd", "echo", "git diff", "git log", "git status", "tree", "whoami"]
    unix = [
        "cat",
        "file",
        "head",
        "ls",
        "pwd",
        "stat",
        "tail",
        "uname",
        "wc",
        "which",
    ]

    if is_windows():
        return common + unix + ["dir", "findstr", "more", "type", "ver", "where"]
    return common + unix


def _get_default_denylist() -> list[str]:
    common = ["gdb", "pdb", "passwd"]
    unix = [
        "nano",
        "vim",
        "vi",
        "emacs",
        "bash -i",
        "sh -i",
        "zsh -i",
        "fish -i",
        "dash -i",
        "screen",
        "tmux",
    ]

    if is_windows():
        return common + unix + [
            "cmd /k",
            "powershell -NoExit",
            "pwsh -NoExit",
            "notepad",
        ]
    return common + unix


def _get_default_denylist_standalone() -> list[str]:
    common = ["python", "python3", "ipython"]
    unix = ["bash", "sh", "nohup", "vi", "vim", "emacs", "nano", "su"]

    if is_windows():
        return common + unix + ["cmd", "powershell", "pwsh", "notepad"]
    return common + unix


_PATH_COMMANDS = {
    "cat",
    "cd",
    "chmod",
    "chown",
    "cp",
    "head",
    "ls",
    "mkdir",
    "mv",
    "rm",
    "stat",
    "tail",
    "touch",
    "wc",
}


def _collect_outside_dirs(command_parts: list[str]) -> set[str]:
    """Collect parent directories referenced outside the workdir.

    Iterates file-manipulating commands (see _PATH_COMMANDS) and inspects
    their arguments as candidate paths. Skips flags (-r, --recursive) and
    chmod mode strings (+x). For any argument that resolves outside the current
    working directory, adds the parent directory (or the path itself when it is
    a directory) to the result set — suitable for building an OUTSIDE_DIRECTORY
    RequiredPermission.
    """
    dirs: set[str] = set()
    for part in command_parts:
        tokens = part.split()
        command = tokens[0] if tokens else None
        if not command or command not in _PATH_COMMANDS:
            continue
        for token in tokens[1:]:
            # Skip CLI flags like -r, --recursive
            if token.startswith("-"):
                continue
            # Skip chmod mode strings like +x, +rwx — they are not file paths
            if command == "chmod" and token.startswith("+"):
                continue
            # Expand env vars with privibe's own environment (the same one the
            # spawned shell inherits) so `$HOME/...` is judged by where it
            # actually points instead of resolving as a relative path under
            # the workdir.
            token = os.path.expandvars(token)
            # Only consider tokens that look like paths
            if not (
                token.startswith(os.sep)
                or token.startswith("~")
                or token.startswith(".")
                or os.sep in token
            ):
                continue
            if is_path_within_workdir(token):
                continue
            # Resolve relative / home-relative / cross-dialect paths, then collect parent dir
            resolved = normalize_tool_path(token).resolve()
            # For a directory target use the dir itself; for a file use its parent
            parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
            dirs.add(parent)
    return dirs


def _iter_candidate_tokens(command_parts: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (raw, expanded) candidate path tokens from every command.

    Every non-flag token is a candidate (env vars expanded with privibe's own
    environment, which the spawned shell inherits); for `--flag=value` tokens
    the value is the candidate.
    """
    for part in command_parts:
        for raw in part.split():
            token = raw.strip("'\"`;|&()")
            if token.startswith("-"):
                if "=" not in token:
                    continue
                token = token.split("=", 1)[1]
            if not token:
                continue
            yield raw, os.path.expandvars(token)


def _find_protected_token(
    command_parts: list[str], protected_paths: list[str]
) -> str | None:
    """Return the first token in any command that resolves into a protected path.

    Deliberately liberal — a stray word that happens to name a protected entry
    costs a refusal, a missed one costs the file.
    """
    if not protected_paths:
        return None
    for raw, expanded in _iter_candidate_tokens(command_parts):
        if is_protected_path(expanded, protected_paths):
            return raw
    return None


def _collect_escalated_outside_dirs(
    command_parts: list[str], exempt: list[str]
) -> set[str]:
    """Collect parent dirs of path-shaped tokens resolving outside the workdir.

    Only tokens that look like paths are judged — a bare word resolves under
    cwd and cannot escape it. Tokens matching an exempt entry are skipped
    (they fall back to the ordinary outside-workdir ASK handling). The result
    feeds escalated OUTSIDE_DIRECTORY permissions: asks that must reach a
    human even under auto_approve.
    """
    dirs: set[str] = set()
    for _raw, expanded in _iter_candidate_tokens(command_parts):
        if not (
            expanded.startswith(("~", ".."))
            or "/" in expanded
            or "\\" in expanded
        ):
            continue
        if is_path_within_workdir(expanded):
            continue
        if is_protected_path(expanded, exempt):
            continue
        resolved = normalize_tool_path(expanded).resolve()
        parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
        dirs.add(parent)
    return dirs


class BashToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    max_output_bytes: int = Field(
        default=16_000, description="Maximum bytes to capture from stdout and stderr."
    )
    default_timeout: int = Field(
        default=300, description="Default timeout for commands in seconds."
    )
    allowlist: list[str] = Field(
        default_factory=_get_default_allowlist,
        description="Command prefixes that are automatically allowed",
    )
    denylist: list[str] = Field(
        default_factory=_get_default_denylist,
        description="Command prefixes that are automatically denied",
    )
    denylist_standalone: list[str] = Field(
        default_factory=_get_default_denylist_standalone,
        description="Commands that are denied only when run without arguments",
    )
    sensitive_patterns: list[str] = Field(
        default=["sudo"],
        description="Command prefixes that always ASK regardless of arity approval.",
    )
    single_call_keywords: list[str] = Field(
        default_factory=lambda: [
            "expensive_call_to_copilot_that_should_be_done_once.sh"
        ],
        description=(
            "Bash commands containing the same keyword run at most once per "
            "assistant message: if several bash calls in one batch share a "
            "keyword, only the first executes and the rest return a skip message "
            "to the model. Use for expensive commands, such as sub-agent or "
            "copilot invocations, that the model tends to issue twice. Matching "
            "is a case-insensitive substring check. The default entry is an "
            "inert self-documenting example that matches nothing real."
        ),
    )
    bash_search_paths: list[str] = Field(
        default_factory=lambda: list(DEFAULT_BASH_SEARCH_PATHS),
        description=(
            "Windows only: locations checked (in order) for Git Bash. Entries "
            "may be a bash.exe path or a Git install directory. Checked before "
            "auto-detection (git on PATH, then `which bash`). Set to [] to "
            "skip the static search and rely on auto-detection only; remove "
            "the field to restore the built-in defaults."
        ),
    )


class BashArgs(BaseModel):
    command: str


class BashResult(BaseModel):
    command: str
    stdout: str
    stderr: str
    returncode: int


class Bash(
    BaseTool[BashArgs, BashResult, BashToolConfig, BaseToolState],
    ToolUIData[BashArgs, BashResult],
):
    description: ClassVar[str] = "Run a one-off bash command and capture its output."

    @classmethod
    def format_call_display(cls, args: BashArgs) -> ToolCallDisplay:
        return ToolCallDisplay(summary=f"bash: {args.command}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, BashResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        return ToolResultDisplay(success=True, message=f"Ran {event.result.command}")

    @classmethod
    def get_status_text(cls) -> str:
        return "Running command"

    def _windows_bash(self) -> str | None:
        """Resolved Git Bash path on Windows; None on Unix or when not found."""
        if not is_windows():
            return None
        return resolve_windows_bash(self.config.bash_search_paths)

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:  # noqa: PLR0911, PLR0912
        if is_windows() and self._windows_bash() is None:
            # cmd.exe fallback: commands are cmd syntax, which the bash
            # grammar can't parse — skip fine-grained resolution and let the
            # blanket ASK permission apply.
            return None

        command_parts = _extract_commands(args.command)
        if not command_parts:
            return None

        def _matches_pattern(command: str, pattern: str) -> bool:
            return command == pattern or command.startswith(pattern + " ")

        def find_denylist_match(command: str) -> str | None:
            return next(
                (p for p in self.config.denylist if _matches_pattern(command, p)), None
            )

        def is_standalone_denylisted(command: str) -> bool:
            parts = command.split()
            if not parts:
                return False
            base_command = parts[0]
            if len(parts) == 1:
                command_name = os.path.basename(base_command)
                if command_name in self.config.denylist_standalone:
                    return True
                if base_command in self.config.denylist_standalone:
                    return True
            return False

        def is_allowlisted(command: str) -> bool:
            return any(
                _matches_pattern(command, pattern) for pattern in self.config.allowlist
            )

        def is_sensitive(command: str) -> bool:
            tokens = command.split()
            if not tokens:
                return False
            return tokens[0] in self.config.sensitive_patterns

        for part in command_parts:
            if matched := find_denylist_match(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' matches denylist pattern '{matched}'. Do not attempt to run this command.",
                )
            if is_standalone_denylisted(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' is not allowed as a standalone command. Do not attempt to run this command.",
                )

        if hit := _find_protected_token(command_parts, self.config.protected_paths):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason=f"Command denied: '{hit}' refers to a protected path. Do not attempt to access it.",
            )

        escalated_dirs: set[str] = set()
        if self.config.protect_outside_workdir:
            escalated_dirs = _collect_escalated_outside_dirs(
                command_parts, self.config.outside_workdir_exempt
            )

        if not escalated_dirs:
            if self.config.permission == ToolPermission.ALWAYS:
                return PermissionContext(permission=ToolPermission.ALWAYS)

        has_sensitive = any(is_sensitive(part) for part in command_parts)
        all_allowlisted = not has_sensitive and all(
            is_allowlisted(part) for part in command_parts
        )
        outside_dirs = _collect_outside_dirs(command_parts)

        if all_allowlisted and not outside_dirs and not escalated_dirs:
            return PermissionContext(permission=ToolPermission.ALWAYS)

        required: list[RequiredPermission] = []
        seen_session: set[str] = set()

        for part in command_parts:
            if not part:
                continue
            tokens = part.split()
            if not tokens:
                continue
            if not is_sensitive(part) and is_allowlisted(part):
                continue

            if is_sensitive(part):
                required.append(
                    RequiredPermission(
                        scope=PermissionScope.COMMAND_PATTERN,
                        invocation_pattern=part,
                        session_pattern=part,
                        label=part,
                    )
                )
            else:
                session_pat = build_session_pattern(tokens)
                if session_pat not in seen_session:
                    seen_session.add(session_pat)
                    required.append(
                        RequiredPermission(
                            scope=PermissionScope.COMMAND_PATTERN,
                            invocation_pattern=part,
                            session_pattern=session_pat,
                            label=session_pat,
                        )
                    )

        if outside_dirs or escalated_dirs:
            escalated_globs = {str(Path(d) / "*") for d in escalated_dirs}
            globs = sorted(
                {str(Path(d) / "*") for d in outside_dirs} | escalated_globs
            )
            for glob in globs:
                required.append(
                    RequiredPermission(
                        scope=PermissionScope.OUTSIDE_DIRECTORY,
                        invocation_pattern=glob,
                        session_pattern=glob,
                        label=f"outside workdir ({glob})",
                        escalated=glob in escalated_globs,
                    )
                )

        if not required:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required
        )

    @final
    def _build_timeout_error(self, command: str, timeout: int) -> ToolError:
        return ToolError(f"Command timed out after {timeout}s: {command!r}")

    @final
    def _build_result(
        self, *, command: str, stdout: str, stderr: str, returncode: int
    ) -> BashResult:
        if returncode != 0:
            error_msg = f"Command failed: {command!r}\n"
            error_msg += f"Return code: {returncode}"
            if stderr:
                error_msg += f"\nStderr: {stderr}"
            if stdout:
                error_msg += f"\nStdout: {stdout}"
            raise ToolError(error_msg.strip())

        return BashResult(
            command=command, stdout=stdout, stderr=stderr, returncode=returncode
        )

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | BashResult, None]:
        timeout = self.config.default_timeout
        max_bytes = self.config.max_output_bytes

        proc = None
        windows_bash = self._windows_bash()
        try:
            # start_new_session is Unix-only, on Windows it's ignored
            kwargs: dict[Literal["start_new_session"], bool] = (
                {} if is_windows() else {"start_new_session": True}
            )

            if windows_bash is not None:
                # Git Bash on Windows. exec + "-c" rather than shell mode with
                # executable=: in shell mode on Windows Python passes `/c`,
                # which bash does not understand.
                proc = await asyncio.create_subprocess_exec(
                    windows_bash,
                    "-c",
                    args.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=_get_base_env(windows_bash=True),
                    **kwargs,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    args.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=_get_base_env(),
                    executable=_get_shell_executable(),
                    **kwargs,
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except TimeoutError:
                await _kill_process_tree(proc)
                raise self._build_timeout_error(args.command, timeout)

            encoding = _get_subprocess_encoding(windows_bash=windows_bash is not None)
            stdout = (
                stdout_bytes.decode(encoding, errors="replace")[:max_bytes]
                if stdout_bytes
                else ""
            )
            stderr = (
                stderr_bytes.decode(encoding, errors="replace")[:max_bytes]
                if stderr_bytes
                else ""
            )

            returncode = proc.returncode or 0

            yield self._build_result(
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
            )

        except (ToolError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise ToolError(f"Error running command {args.command!r}: {exc}") from exc
        finally:
            if proc is not None:
                await _kill_process_tree(proc)
