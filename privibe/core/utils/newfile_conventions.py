"""What a new file should look like, decided before the model has to think.

The ladder, highest wins:

1. ``.gitattributes`` ``eol`` for the path (``git check-attr``).
2. ``.editorconfig``: ``end_of_line``, ``charset``, ``indent_style``,
   ``indent_size``/``tab_width``, ``insert_final_newline``,
   ``trim_trailing_whitespace``.
3. Extension rule: shell scripts are LF, batch files are CRLF, whatever the
   platform says. A CRLF ``.sh`` dies on its first line.
4. Siblings: the dominant working-tree ending of tracked files in the same
   directory (``git ls-files --eol``), then the whole repo.
5. Platform plus ``core.autocrlf``. ``core.autocrlf`` lives in the machine's
   gitconfig, so it is environment, not a repo decision, and sits at the
   bottom.

A BOM is only ever asked for by ``.editorconfig`` (``charset = utf-8-bom``).
Siblings vote on endings, never on BOMs.

Only run this for files that do not exist yet. Existing files keep what they
have; the ladder has no vote there. ``editorconfig_for`` is the one piece that
also applies to existing files, for the two save-time rules that touch only
the lines the model wrote.

Results are cached per directory for the life of the process; a session does
not need to re-ask git about the same folder four hundred times.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys

_GIT_TIMEOUT = 5.0
_LS_FILES_CAP = 20_000

# Extension rule (step 3). Kept tiny; the repo files above it win.
_LF_EXTENSIONS = {".sh", ".bash", ".zsh", ".ksh", ".fish"}
_CRLF_EXTENSIONS = {".bat", ".cmd"}


@dataclass(frozen=True)
class EditorConfig:
    end_of_line: str | None = None  # "\n" | "\r\n" | "\r"
    charset: str | None = None  # "utf-8" | "utf-8-bom" | "latin1" | ...
    indent_style: str | None = None  # "tab" | "space"
    indent_size: int | None = None
    tab_width: int | None = None
    insert_final_newline: bool | None = None
    trim_trailing_whitespace: bool | None = None

    @property
    def wants_bom(self) -> bool:
        return self.charset == "utf-8-bom"

    @property
    def python_encoding(self) -> str | None:
        if self.charset in {None, "utf-8", "utf-8-bom"}:
            return "utf-8" if self.charset else None
        if self.charset == "latin1":
            return "latin-1"
        if self.charset in {"utf-16be", "utf-16le"}:
            return self.charset
        return self.charset


@dataclass(frozen=True)
class Conventions:
    newline: str
    encoding: str
    bom: bool
    final_newline: bool
    trim_trailing_whitespace: bool
    indent_style: str | None
    indent_size: int | None
    source: str  # which rung decided the newline, for notes and tests


# ---------------------------------------------------------------------------
# .editorconfig
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^\s*\[(.*)\]\s*$")
_KV_RE = re.compile(r"^\s*([^=:#;]+?)\s*[=:]\s*(.*?)\s*$")


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """EditorConfig section glob to a regex over a forward-slash path.

    Patterns without a slash match the basename anywhere; patterns with one are
    anchored at the .editorconfig's directory. Supports ``*``, ``**``, ``?``,
    ``[...]`` and ``{a,b}``.
    """
    anchored = "/" in pattern
    if pattern.startswith("/"):
        pattern = pattern[1:]
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j
        elif c == "{":
            j = pattern.find("}", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                alts = pattern[i + 1 : j].split(",")
                out.append("(?:" + "|".join(re.escape(a) for a in alts) + ")")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    body_re = "".join(out)
    if anchored:
        return re.compile(f"^{body_re}$")
    return re.compile(f"(?:^|/){body_re}$")


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


@lru_cache(maxsize=256)
def _parse_editorconfig(path: Path) -> tuple[bool, tuple[tuple[re.Pattern[str], tuple[tuple[str, str], ...]], ...]]:
    """(is_root, sections) for one .editorconfig file; cached by path."""
    is_root = False
    sections: list[tuple[re.Pattern[str], list[tuple[str, str]]]] = []
    current: list[tuple[str, str]] | None = None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, ()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = []
            sections.append((_glob_to_regex(m.group(1).strip()), current))
            continue
        kv = _KV_RE.match(line)
        if not kv:
            continue
        key, value = kv.group(1).strip().lower(), kv.group(2).strip()
        if current is None:
            if key == "root":
                is_root = value.lower() == "true"
            continue
        current.append((key, value))
    return is_root, tuple((p, tuple(kvs)) for p, kvs in sections)


def editorconfig_for(path: Path) -> EditorConfig:
    """Effective .editorconfig properties for ``path`` (may not exist yet)."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    directory = path.parent
    chain: list[Path] = []
    for d in [directory, *directory.parents]:
        cfg = d / ".editorconfig"
        if cfg.is_file():
            chain.append(cfg)
            is_root, _ = _parse_editorconfig(cfg)
            if is_root:
                break
    props: dict[str, str] = {}
    # Farthest ancestor first; nearer files override.
    for cfg in reversed(chain):
        _, sections = _parse_editorconfig(cfg)
        try:
            rel = path.relative_to(cfg.parent).as_posix()
        except ValueError:
            continue
        for pattern, kvs in sections:
            if pattern.search(rel):
                for key, value in kvs:
                    props[key] = value
    return _editorconfig_from_props(props)


def _editorconfig_from_props(props: dict[str, str]) -> EditorConfig:
    def as_int(key: str) -> int | None:
        v = props.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    eol_map = {"lf": "\n", "crlf": "\r\n", "cr": "\r"}
    eol = props.get("end_of_line", "").lower()
    charset = props.get("charset", "").lower() or None
    indent_style = props.get("indent_style", "").lower() or None
    if indent_style not in {None, "tab", "space"}:
        indent_style = None
    indent_size: int | None
    if props.get("indent_size", "").lower() == "tab":
        indent_size = as_int("tab_width")
    else:
        indent_size = as_int("indent_size")
    return EditorConfig(
        end_of_line=eol_map.get(eol),
        charset=charset,
        indent_style=indent_style,
        indent_size=indent_size,
        tab_width=as_int("tab_width") or indent_size,
        insert_final_newline=_parse_bool(props.get("insert_final_newline", "")),
        trim_trailing_whitespace=_parse_bool(props.get("trim_trailing_whitespace", "")),
    )


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@lru_cache(maxsize=256)
def _git_toplevel(directory: Path) -> Path | None:
    out = _git(["rev-parse", "--show-toplevel"], directory)
    if not out:
        return None
    return Path(out.strip())


def gitattributes_eol(path: Path) -> str | None:
    """The eol attribute git applies to ``path``, or None when unspecified."""
    directory = path.parent
    if _git_toplevel(directory) is None:
        return None
    out = _git(["check-attr", "eol", "--", path.name], directory)
    if not out:
        return None
    # "<path>: eol: lf"
    value = out.strip().rsplit(":", 1)[-1].strip().lower()
    return {"lf": "\n", "crlf": "\r\n"}.get(value)


@lru_cache(maxsize=256)
def _ls_files_eol(directory: Path) -> tuple[tuple[str, str], ...]:
    """(path, working-tree eol) for tracked files under ``directory``."""
    top = _git_toplevel(directory)
    if top is None:
        return ()
    out = _git(["ls-files", "--eol", "-z", "--", "."], directory)
    if not out:
        return ()
    entries: list[tuple[str, str]] = []
    for record in out.split("\0"):
        if not record:
            continue
        head, _, rel = record.partition("\t")
        w = next((tok[2:] for tok in head.split() if tok.startswith("w/")), "")
        entries.append((rel, w))
        if len(entries) >= _LS_FILES_CAP:
            break
    return tuple(entries)


def sibling_ending(path: Path) -> str | None:
    """Dominant working-tree ending of tracked siblings, then the subtree."""
    directory = path.parent
    entries = _ls_files_eol(directory)
    if not entries:
        return None

    def vote(candidates: list[str]) -> str | None:
        counts = Counter(c for c in candidates if c in {"lf", "crlf"})
        if not counts:
            return None
        best = max(counts.values())
        winners = {k for k, v in counts.items() if v == best}
        chosen = "lf" if "lf" in winners else "crlf"
        return "\n" if chosen == "lf" else "\r\n"

    same_dir = [w for rel, w in entries if "/" not in rel]
    return vote(same_dir) or vote([w for _, w in entries])


@lru_cache(maxsize=64)
def _autocrlf(directory: Path) -> str:
    out = _git(["config", "--get", "core.autocrlf"], directory)
    return (out or "").strip().lower()


def platform_ending(directory: Path) -> str:
    if sys.platform == "win32" and _autocrlf(directory) == "true":
        return "\r\n"
    return "\n"


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def extension_ending(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in _LF_EXTENSIONS:
        return "\n"
    if suffix in _CRLF_EXTENSIONS:
        return "\r\n"
    return None


def conventions_for(path: Path) -> Conventions:
    """Everything a writer needs to create ``path`` the way its repo expects."""
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    ec = editorconfig_for(path)

    newline: str | None
    source: str
    if (newline := gitattributes_eol(path)) is not None:
        source = ".gitattributes"
    elif ec.end_of_line is not None:
        newline, source = ec.end_of_line, ".editorconfig"
    elif (newline := extension_ending(path)) is not None:
        source = "extension"
    elif (newline := sibling_ending(path)) is not None:
        source = "siblings"
    else:
        newline, source = platform_ending(path.parent), "platform"

    return Conventions(
        newline=newline,
        encoding=ec.python_encoding or "utf-8",
        bom=ec.wants_bom,
        final_newline=True if ec.insert_final_newline is None else ec.insert_final_newline,
        trim_trailing_whitespace=bool(ec.trim_trailing_whitespace),
        indent_style=ec.indent_style,
        indent_size=ec.indent_size,
        source=source,
    )


def clear_caches() -> None:
    """Forget every cached git and .editorconfig lookup. Test isolation hook."""
    _parse_editorconfig.cache_clear()
    _git_toplevel.cache_clear()
    _ls_files_eol.cache_clear()
    _autocrlf.cache_clear()
