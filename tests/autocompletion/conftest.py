from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def force_walk_enumeration(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the file enumerator to its walk backend for autocompletion tests.

    These tests build throwaway directory trees and assert on completion results.
    The walk backend depends only on the filesystem, so results are deterministic
    regardless of whether git/ripgrep are installed or how global git config is
    set. The git and ripgrep backends get their own coverage in
    test_file_enumerator.py, where individual tests opt back in.
    """
    monkeypatch.setenv("PRIVIBE_FILE_ENUMERATOR", "walk")
    yield
