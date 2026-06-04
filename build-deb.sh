#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DATE="$(date +%y%m%d%H%M)"

BASE_VERSION="$(grep '^version' pyproject.toml | head -1 | sed 's/.*= *"\(.*\)"/\1/')"
VERSION="${BASE_VERSION}.${BUILD_DATE}"
ARCH="$(dpkg --print-architecture)"
PKG_NAME="fws-privibe"
DEB_NAME="${PKG_NAME}_${VERSION}_${ARCH}.deb"

echo "==> Building ${DEB_NAME}"

# ── 0. Generate build-info file (gitignored, picked up by privibe/__init__.py) ─
cat > privibe/_build_info.py <<EOF
BUILD_STAMP = "${BUILD_DATE}"
EOF

PKG_DIR=""
cleanup() {
    rm -f "${SCRIPT_DIR}/privibe/_build_info.py"
    if [ -n "${PKG_DIR}" ] && [ -d "${PKG_DIR}" ]; then
        rm -rf "${PKG_DIR}"
    fi
}
trap cleanup ERR INT TERM EXIT

# ── 1. Install build dependencies ────────────────────────────────────────────
echo "==> Installing build dependencies..."
uv sync --group build

# ── 2. Build standalone binary with PyInstaller ──────────────────────────────
echo "==> Running PyInstaller..."
uv run --group build pyinstaller privibe.spec --clean --noconfirm

BINARY="dist/privibe"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: PyInstaller output not found at $BINARY" >&2
    exit 1
fi

# ── 3. Assemble .deb package structure ───────────────────────────────────────
echo "==> Assembling package structure..."
PKG_DIR="$(mktemp -d)"

install -d "$PKG_DIR/DEBIAN"
install -d "$PKG_DIR/usr/bin"
install -m 755 "$BINARY" "$PKG_DIR/usr/bin/privibe"

cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Architecture: ${ARCH}
Maintainer: Privibe
Section: utils
Priority: optional
Description: Privibe CLI coding agent
 A minimal CLI coding agent for private, local-first development.
 Self-contained binary — no Python installation required.
EOF

# ── 4. Build the .deb ────────────────────────────────────────────────────────
echo "==> Building .deb..."
dpkg-deb --build --root-owner-group "$PKG_DIR" "$DEB_NAME"

echo "==> Built: $DEB_NAME"

# ── 4b. Create/update latest symlink ─────────────────────────────────────────
LATEST_NAME="${PKG_NAME}_latest_${ARCH}.deb"
ln -sf "$DEB_NAME" "$LATEST_NAME"
echo "==> Symlink: $LATEST_NAME -> $DEB_NAME"

# ── 5. Install ───────────────────────────────────────────────────────────────
echo "==> Installing..."
sudo dpkg -i "$DEB_NAME"

echo ""
echo "Done. Run: privibe"
