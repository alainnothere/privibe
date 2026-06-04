# Running Privibe in Windows (Git Bash)

There is no native Windows installer (the repo only ships `.deb` and Nix). But Privibe is a normal Python package, so you can run it on Windows from Git Bash with `uv`.

## 1. Build a source zip on the Linux side

Use `build-windows-zip.sh` at the repo root. It works from any directory:

```bash
~/theOtherDocs/gitSrc/privibe/build-windows-zip.sh
```

What it does:

- Anchors on its own location, not the caller's CWD.
- Uses `git archive` so only tracked files are shipped — no `.venv`, `.git`, `.deb`s, `dist/`, `build/`, or scratch files.
- `git stash create` snapshots uncommitted edits to tracked files (e.g. a modified `uv.lock`) so they make it into the zip.
- Drops a `WINDOWS-INSTALL.txt` quickstart inside the archive.
- Output: `privibe-source-YYMMDDHHMM.zip` in the repo root.

Copy that zip to the Windows machine.

## 2. Prerequisites on Windows

- **Git Bash** — https://git-scm.com/download/win
- **uv** — https://docs.astral.sh/uv/getting-started/installation/ (uv will fetch Python 3.12 itself if needed)

Python 3.12+ is required (`pyproject.toml` `requires-python = ">=3.12"`). All native deps (`sounddevice`, `tree-sitter`, `tree-sitter-bash`, `pyperclip`) ship Windows wheels — no compiler needed.

## 3. Install

Unzip somewhere stable, e.g. `~/bin/privibe`. Then in Git Bash, pick one:

### Option A — Local venv inside the unzipped folder

```bash
cd ~/bin/privibe
uv sync
uv run privibe
```

After `uv sync` the folder is self-contained. Don't ever copy a `.venv` from Linux/macOS to Windows — venvs are not cross-platform; build it on Windows.

To reset:

```bash
rm -rf .venv && uv sync
```

### Option B — `uv tool install` (recommended)

Lets you call plain `privibe` from any directory without the indirection.

```bash
uv tool install ~/bin/privibe
# update later:
uv tool install --reinstall ~/bin/privibe
```

uv prints the bin path on first run (typically `%USERPROFILE%\.local\bin`); make sure that's on `PATH`.

## 4. Run privibe against a project in a different directory

privibe operates on the **current working directory** at launch — not where it's installed. From any folder you want privibe to work on, use one of these:

```bash
# A. Point uv run at the install (CWD stays where you cd'd)
cd /a/b/c
uv run --project ~/bin/privibe privibe

# B. Call the venv binary directly (Windows venvs use Scripts/, not bin/)
~/bin/privibe/.venv/Scripts/privibe

# C. Add the venv's Scripts dir to PATH (in ~/.bashrc)
export PATH="$HOME/bin/privibe/.venv/Scripts:$PATH"

# D. With `uv tool install` from Option B above, just:
privibe
```

A handy alias for option B in `~/.bashrc`:

```bash
alias privibe='~/bin/privibe/.venv/Scripts/privibe'
```

## 5. Config directory: `~/.privibe`

Default config dir is `~/.privibe` — defined in `privibe/core/paths/_vibe_home.py`. Override with the `VIBE_HOME` env var:

```bash
export VIBE_HOME="$HOME/.privibe"   # or any absolute path
privibe
```

In Git Bash, `~` is `C:\Users\<you>`. `Path.home()` resolves via `USERPROFILE` and should match — verify with:

```bash
python -c "from pathlib import Path; print(Path.home() / '.privibe')"
```

Layout under `~/.privibe`:

```
~/.privibe/
├── config.toml
├── .env
├── skills/      # user-level skills
├── tools/       # user-level custom tools
├── agents/      # user-level subagents
├── prompts/     # user-level prompt overrides
├── plans/
└── logs/
      ├── privibe.log
    └── session/
```

## 6. Skills layout

Each skill is a **subdirectory** containing a `SKILL.md` (loose `.md` files at the top of `skills/` are silently ignored):

```
~/.privibe/skills/
├── my-first-skill/
│   └── SKILL.md
└── another-skill/
    └── SKILL.md
```

`SKILL.md` requires YAML frontmatter; the `name:` field must match the directory name:

```markdown
---
name: my-first-skill
description: One-line description used to decide when to invoke this
---

Body of the skill...
```

Project-level skills (`<project>/.privibe/skills/<name>/SKILL.md`) only load when the project folder is **trusted** in privibe. User-level skills in `~/.privibe/skills/` have no trust requirement.

## 7. Diagnostics

On startup, privibe logs to `~/.privibe/logs/privibe.log`:

- `Discovered N skill(s) from M search path(s)` — confirms the skills directory was found.
- `Failed to parse skill at <path>: <reason>` — frontmatter or naming problem.

If neither appears, privibe isn't looking where you think — re-check `Path.home()` and `VIBE_HOME`.

## 8. Caveats

- The agent's bash tool shells out to `bash` — Git Bash provides one, so it works. Pure `cmd.exe`/PowerShell will hit edge cases.
- The `.deb` files in the repo are Linux-only; ignore them on Windows.
- For the smoothest experience overall, consider WSL2 (Ubuntu) and `sudo dpkg -i fws-privibe_latest_amd64.deb`.
