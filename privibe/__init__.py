from __future__ import annotations

from pathlib import Path

VIBE_ROOT = Path(__file__).parent

_BASE_VERSION = "0.1.0"
try:
    from privibe._build_info import BUILD_STAMP

    __version__ = f"{_BASE_VERSION}.{BUILD_STAMP}"
except ImportError:
    __version__ = _BASE_VERSION
