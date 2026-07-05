from __future__ import annotations

from privibe.core.rewind.diffing import build_file_diff


def _b(text: str) -> bytes:
    return text.encode("utf-8")


def _classes(diff) -> list[str]:
    return [cls for hunk in diff.hunks for cls, _ in hunk]


def _texts(diff) -> list[str]:
    return [text for hunk in diff.hunks for _, text in hunk]


def test_noop_returns_none():
    assert build_file_diff("f.py", _b("a\nb\n"), _b("a\nb\n")) is None


def test_simple_edit_has_red_and_green():
    old = _b("a\nb\nc\n")
    new = _b("a\nB\nc\n")
    diff = build_file_diff("f.py", old, new)
    assert diff is not None
    assert diff.kind == "diff"
    classes = _classes(diff)
    assert "diff-removed" in classes
    assert "diff-added" in classes
    # The changed lines carry their +/- prefix.
    assert "-b" in _texts(diff)
    assert "+B" in _texts(diff)


def test_create_is_all_green():
    diff = build_file_diff("f.py", None, _b("x\ny\n"))
    assert diff is not None
    assert diff.kind == "diff"
    classes = _classes(diff)
    assert "diff-added" in classes
    assert "diff-removed" not in classes


def test_delete_is_all_red():
    diff = build_file_diff("f.py", _b("x\ny\n"), None)
    assert diff is not None
    classes = _classes(diff)
    assert "diff-removed" in classes
    assert "diff-added" not in classes


def test_long_run_is_cropped_to_head_tail():
    # 9 removed lines -> 2 + "5 more lines cut" + 2.
    old = "\n".join(f"old{i}" for i in range(9)) + "\n"
    new = "totally\ndifferent\n"
    diff = build_file_diff("f.py", _b(old), _b(new))
    assert diff is not None
    texts = _texts(diff)
    assert any("5 more lines cut" in t for t in texts)
    # Only 4 of the 9 old lines survive (first 2 + last 2).
    removed = [
        text for hunk in diff.hunks for cls, text in hunk if cls == "diff-removed"
    ]
    assert len(removed) == 4
    assert "-old0" in removed and "-old8" in removed


def test_short_run_not_cropped():
    old = "\n".join(f"old{i}" for i in range(4)) + "\n"
    new = "new\n"
    diff = build_file_diff("f.py", _b(old), _b(new))
    assert diff is not None
    assert not any("more lines cut" in t for t in _texts(diff))


def test_binary_old_side_detected():
    diff = build_file_diff("f.bin", b"a\x00b", _b("text\n"))
    assert diff is not None
    assert diff.kind == "binary"
    assert diff.note == "Binary file - no diff shown"
    assert diff.hunks == []


def test_binary_new_side_detected():
    diff = build_file_diff("f.bin", _b("text\n"), b"\x00\x01\x02")
    assert diff is not None
    assert diff.kind == "binary"


def test_large_file_falls_back_to_sample():
    old = "\n".join(f"line{i}" for i in range(20_001)) + "\n"
    new = "\n".join(f"changed{i}" for i in range(20_001)) + "\n"
    diff = build_file_diff("big.py", _b(old), _b(new), tool_name="hashed_replace_block")
    assert diff is not None
    assert diff.kind == "sample"
    header = diff.hunks[0][0][1]
    assert "preview only" in header
    assert any("more lines cut" in t for _, t in diff.hunks[0])


def test_sample_write_file_header_says_entire_file():
    old = "\n".join(f"line{i}" for i in range(20_001)) + "\n"
    new = "\n".join(f"changed{i}" for i in range(20_001)) + "\n"
    diff = build_file_diff("big.py", _b(old), _b(new), tool_name="write_file")
    assert diff is not None
    assert diff.kind == "sample"
    assert "Replaced entire file" in diff.hunks[0][0][1]


def test_multiple_changes_produce_multiple_hunks():
    old = "\n".join(str(i) for i in range(40)) + "\n"
    lines = [str(i) for i in range(40)]
    lines[2] = "TWO"
    lines[30] = "THIRTY"
    new = "\n".join(lines) + "\n"
    diff = build_file_diff("f.py", _b(old), _b(new))
    assert diff is not None
    assert diff.kind == "diff"
    # Two well-separated edits -> two hunks.
    assert len(diff.hunks) == 2
