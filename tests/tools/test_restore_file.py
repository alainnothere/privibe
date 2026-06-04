from __future__ import annotations

import pytest

from privibe.core.rewind.manager import FileSnapshot
from privibe.core.rewind.undo_stack import FileUndoStack, canonical_key
from privibe.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
)
from privibe.core.tools.builtins.restore_file import (
    RestoreFile,
    RestoreFileArgs,
    RestoreFileConfig,
    RestoreFileResult,
)
from tests.mock.utils import collect_result


@pytest.fixture
def restore_tool():
    return RestoreFile(config=RestoreFileConfig(), state=BaseToolState())


def _ctx_with_stack(stack: FileUndoStack) -> InvokeContext:
    return InvokeContext(tool_call_id="call_test", undo_stack=stack)


def _snap(path: str, content: bytes | None) -> FileSnapshot:
    return FileSnapshot(path=canonical_key(path), content=content)


@pytest.mark.asyncio
async def test_restore_reverts_content(tmp_path, monkeypatch, restore_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.txt"
    f.write_text("broken edit")
    stack = FileUndoStack()
    stack.capture(_snap(str(f), b"good original"))

    result = await collect_result(
        restore_tool.run(RestoreFileArgs(path="f.txt"), _ctx_with_stack(stack))
    )

    assert isinstance(result, RestoreFileResult)
    assert result.action == "restored"
    assert result.remaining == 0
    assert f.read_text() == "good original"


@pytest.mark.asyncio
async def test_restore_deletes_created_file(tmp_path, monkeypatch, restore_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "created.txt"
    stack = FileUndoStack()
    # Capture before the create (file absent), then the edit creates it.
    stack.capture(_snap(str(f), None))
    f.write_text("whoops")

    result = await collect_result(
        restore_tool.run(RestoreFileArgs(path="created.txt"), _ctx_with_stack(stack))
    )

    assert result.action == "deleted"
    assert not f.exists()


@pytest.mark.asyncio
async def test_restore_with_no_version_raises(tmp_path, monkeypatch, restore_tool):
    monkeypatch.chdir(tmp_path)
    stack = FileUndoStack()
    with pytest.raises(ToolError, match="No restore point"):
        await collect_result(
            restore_tool.run(RestoreFileArgs(path="nope.txt"), _ctx_with_stack(stack))
        )


@pytest.mark.asyncio
async def test_restore_without_context_raises(restore_tool):
    with pytest.raises(ToolError, match="not available"):
        await collect_result(restore_tool.run(RestoreFileArgs(path="f.txt"), None))


@pytest.mark.asyncio
async def test_restore_empty_path_raises(restore_tool):
    stack = FileUndoStack()
    with pytest.raises(ToolError, match="empty"):
        await collect_result(
            restore_tool.run(RestoreFileArgs(path="   "), _ctx_with_stack(stack))
        )


def test_restore_tool_is_not_captured(restore_tool):
    # restore_file must not push a version (it consumes the stack), so it must
    # not participate in the pre-edit capture hook.
    assert restore_tool.get_file_snapshot(RestoreFileArgs(path="f.txt")) is None


def test_restore_tool_is_marked_mutating():
    # It writes to disk, so the scheduler must serialize it with other writes.
    assert RestoreFile.mutates_files is True
