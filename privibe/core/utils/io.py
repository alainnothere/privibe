from __future__ import annotations

from pathlib import Path

import anyio

# Fallback for non-UTF-8 files. Explicit, never the locale: on Linux the
# locale is UTF-8 and a locale fallback would fail the same way twice, on
# Windows it is cp1252 anyway. cp1252 is what legacy Windows-authored source
# files actually are; a BOM-less UTF-8 file never reaches this path.
FALLBACK_ENCODING = "cp1252"


def read_safe(path: Path, *, raise_on_error: bool = False) -> str:
    """Read a text file trying UTF-8 first (BOM tolerated and stripped),
    falling back to cp1252.

    On fallback, undecodable bytes are replaced with U+FFFD (REPLACEMENT CHARACTER).
    When raise_on_error is True, decode errors propagate.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, ValueError):
        if raise_on_error:
            return path.read_text(encoding=FALLBACK_ENCODING)
        return path.read_text(encoding=FALLBACK_ENCODING, errors="replace")


async def read_safe_async(path: Path, *, raise_on_error: bool = False) -> str:
    apath = anyio.Path(path)
    try:
        return await apath.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, ValueError):
        if raise_on_error:
            return await apath.read_text(encoding=FALLBACK_ENCODING)
        return await apath.read_text(encoding=FALLBACK_ENCODING, errors="replace")
