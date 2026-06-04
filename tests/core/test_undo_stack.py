from __future__ import annotations

import pytest

from privibe.core.rewind.manager import FileSnapshot
from privibe.core.rewind.undo_stack import (
    FileUndoStack,
    NothingToRestoreError,
    canonical_key,
)


def _snap(path: str, content: bytes | None) -> FileSnapshot:
    """Build a snapshot keyed exactly as the capture hook would (canonical path)."""
    return FileSnapshot(path=canonical_key(path), content=content)


def test_capture_then_restore_round_trips_content(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("v2 (current)")
    stack = FileUndoStack()

    stack.capture(_snap(str(f), b"v1 (original)"))
    outcome = stack.restore(str(f))

    assert outcome.action == "restored"
    assert outcome.remaining == 0
    assert f.read_text() == "v1 (original)"


def test_walks_backward_one_edit_per_call(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("current")
    stack = FileUndoStack()

    stack.capture(_snap(str(f), b"v0"))
    stack.capture(_snap(str(f), b"v1"))
    stack.capture(_snap(str(f), b"v2"))

    assert stack.restore(str(f)).remaining == 2
    assert f.read_text() == "v2"
    assert stack.restore(str(f)).remaining == 1
    assert f.read_text() == "v1"
    assert stack.restore(str(f)).remaining == 0
    assert f.read_text() == "v0"


def test_restore_of_created_file_deletes_it(tmp_path):
    f = tmp_path / "created.txt"
    stack = FileUndoStack()

    # Capture happens BEFORE the create, so the file is genuinely absent here.
    stack.capture(_snap(str(f), None))
    # The edit then creates the file.
    f.write_text("content the edit wrote")

    outcome = stack.restore(str(f))

    assert outcome.action == "deleted"
    assert not f.exists()


def test_skip_if_same_does_not_push_redundant_version(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("current")
    stack = FileUndoStack()

    stack.capture(_snap(str(f), b"same"))
    stack.capture(_snap(str(f), b"same"))  # identical -> skipped

    stack.restore(str(f))
    # Only one real version existed, so a second restore must fail.
    with pytest.raises(NothingToRestoreError):
        stack.restore(str(f))


def test_read_failure_is_not_recorded_as_a_delete_target(tmp_path):
    # content=None but the file EXISTS on disk => the snapshot helper could not
    # read it, not that it was absent. Recording this would let a later restore
    # delete a real file, so capture must skip it.
    f = tmp_path / "exists.txt"
    f.write_text("real content")
    stack = FileUndoStack()

    stack.capture(_snap(str(f), None))

    assert not stack.has_versions(str(f))
    with pytest.raises(NothingToRestoreError):
        stack.restore(str(f))
    assert f.read_text() == "real content"


def test_genuine_absence_is_recorded(tmp_path):
    # content=None and the path does NOT exist => genuine "did not exist".
    f = tmp_path / "absent.txt"
    stack = FileUndoStack()

    stack.capture(_snap(str(f), None))

    assert stack.has_versions(str(f))


def test_bounded_eviction_keeps_only_most_recent(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("current")
    stack = FileUndoStack(max_versions=3)

    for i in range(5):
        stack.capture(_snap(str(f), f"v{i}".encode()))

    # Oldest (v0, v1) evicted; v2 is the deepest reachable.
    assert stack.restore(str(f)).remaining == 2  # restores v4
    assert stack.restore(str(f)).remaining == 1  # restores v3
    assert stack.restore(str(f)).remaining == 0  # restores v2
    assert f.read_text() == "v2"
    with pytest.raises(NothingToRestoreError):
        stack.restore(str(f))


def test_oversized_entry_is_skipped(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("current")
    stack = FileUndoStack(max_entry_bytes=10)

    stack.capture(_snap(str(f), b"x" * 100))

    assert not stack.has_versions(str(f))


def test_empty_stack_raises(tmp_path):
    stack = FileUndoStack()
    with pytest.raises(NothingToRestoreError):
        stack.restore(str(tmp_path / "never_touched.txt"))


def test_clear_drops_all_versions(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("current")
    stack = FileUndoStack()
    stack.capture(_snap(str(f), b"v0"))

    stack.clear()

    assert not stack.has_versions(str(f))


def test_relative_and_absolute_paths_hit_the_same_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "rel.txt"
    f.write_text("current")
    stack = FileUndoStack()

    # Capture with the absolute (canonical) path the hook would produce...
    stack.capture(_snap(str(f), b"original"))
    # ...and restore using a relative path the model might pass.
    outcome = stack.restore("rel.txt")

    assert outcome.action == "restored"
    assert f.read_text() == "original"


def test_per_file_isolation(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("a-current")
    b.write_text("b-current")
    stack = FileUndoStack()

    stack.capture(_snap(str(a), b"a-old"))
    stack.capture(_snap(str(b), b"b-old"))

    stack.restore(str(a))
    assert a.read_text() == "a-old"
    assert b.read_text() == "b-current"  # untouched
    assert stack.has_versions(str(b))
