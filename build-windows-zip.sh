#!/usr/bin/env bash
# Build a source zip of privibe to copy to a Windows machine.
# On Windows: unzip, then in Git Bash run `uv sync && uv run privibe`.

set -euo pipefail

command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v zip >/dev/null || { echo "zip not found (apt install zip)" >&2; exit 1; }

# Anchor on the script's own location, not the caller's CWD,
# so it works regardless of where you invoke it from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

STAMP="$(date -u +%y%m%d%H%M)"
OUT="${REPO_ROOT}/privibe-source-${STAMP}.zip"

# Snapshot the working tree: tracked files as they are on disk, plus untracked
# files git does not ignore. Done through a scratch index because `git stash
# create` takes only tracked files: a new module imported by a modified one
# shipped without itself once, and the zip crashed on import.
# .gitignore still rules, so .venv, .git, *.deb, dist/, build/ etc. stay out.
TMP_INDEX="$(mktemp)"
cp "$(git rev-parse --git-path index)" "${TMP_INDEX}"
GIT_INDEX_FILE="${TMP_INDEX}" git add -A
SNAPSHOT="$(GIT_INDEX_FILE="${TMP_INDEX}" git write-tree)"
rm -f "${TMP_INDEX}"

git archive --format=zip --prefix=privibe/ -o "${OUT}" "${SNAPSHOT}"

# Drop a Windows quickstart inside the archive.
TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT
mkdir -p "${TMPDIR}/privibe"

# Inject a build-info file so the runtime __version__ knows its stamp.
# privibe/__init__.py picks it up via a try/except ImportError.
mkdir -p "${TMPDIR}/privibe/privibe"
cat > "${TMPDIR}/privibe/privibe/_build_info.py" <<EOF
BUILD_STAMP = "${STAMP}"
EOF
(cd "${TMPDIR}" && zip -qr "${OUT}" privibe/privibe/_build_info.py)

cat > "${TMPDIR}/privibe/WINDOWS-INSTALL.txt" <<'EOF'
privibe - Windows quickstart (Git Bash)

Prerequisites:
  - Git Bash      https://git-scm.com/download/win
  - uv            https://docs.astral.sh/uv/getting-started/installation/
                  (uv will fetch Python 3.12 for you if needed)

Install & run, from this folder in Git Bash:
  uv sync
  uv run privibe

After `uv sync` this folder is self-contained; just `uv run privibe` next time.

Notes:
  - Do NOT copy a .venv from Linux/macOS. Run `uv sync` on Windows.
  - Reset the venv:  rm -rf .venv && uv sync
EOF

(cd "${TMPDIR}" && zip -qr "${OUT}" privibe/WINDOWS-INSTALL.txt)

# Update the "latest" symlink so callers can grab a stable filename
# (mirrors build-deb.sh which maintains fws-privibe_latest_amd64.deb).
LATEST="${REPO_ROOT}/privibe-source-latest.zip"
ln -sfn "$(basename "${OUT}")" "${LATEST}"

echo "Built:  ${OUT}"
echo "Size:   $(du -h "${OUT}" | cut -f1)"
echo "Latest: ${LATEST} -> $(basename "${OUT}")"

cp  "$(basename "${OUT}")" ~/sharedViaLlama.cpp/privibe-source-latest.zip 
