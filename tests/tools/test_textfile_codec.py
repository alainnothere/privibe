"""The byte codec: what it reads it writes back, byte for byte, except the
lines the caller named.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.utils.textfile import (
    BinaryFileError,
    TextFile,
    UnencodableError,
    content_to_lines,
    dominant_ending,
    load_text_file,
    reattach_endings,
    render,
    render_text,
    split_keepends,
    trim_trailing_whitespace,
)

CRLF_BOM = b"\xef\xbb\xbfline one\r\nline two\r\nline three\r\n"
MIXED = b"a\nb\r\nc\r\nd\n"
NO_FINAL = b"x\ny\nz"


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def test_split_keepends_only_real_terminators():
    # \x0c (form feed) and   are content, not line breaks.
    text = "a\x0cb c\r\nd\re\nf"
    assert split_keepends(text) == ["a\x0cb c\r\n", "d\r", "e\n", "f"]


def test_split_keepends_empty():
    assert split_keepends("") == []


def test_dominant_ending_ties_go_to_default():
    assert dominant_ending(["\r\n", "\n"]) == "\n"
    assert dominant_ending(["\r\n", "\r\n", "\n"]) == "\r\n"
    assert dominant_ending([], default="\r\n") == "\r\n"


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


def test_from_bytes_strips_bom_and_normalizes():
    tf = TextFile.from_bytes(CRLF_BOM)
    assert tf.bom is True
    assert tf.encoding == "utf-8"
    assert tf.newline == "\r\n"
    assert tf.lines == ["line one\n", "line two\n", "line three\n"]
    assert tf.endings == ["\r\n", "\r\n", "\r\n"]
    assert tf.had_final_newline is True
    assert "﻿" not in tf.text


def test_from_bytes_mixed_is_flagged():
    tf = TextFile.from_bytes(MIXED)
    assert tf.mixed is True
    assert tf.newline == "\n"  # 2 CRLF vs 2 LF: a tie goes to LF
    assert any("mixed" in n for n in tf.notes)


def test_from_bytes_no_final_newline():
    tf = TextFile.from_bytes(NO_FINAL)
    assert tf.had_final_newline is False
    assert tf.lines == ["x\n", "y\n", "z\n"]
    assert tf.endings == ["\n", "\n", ""]


def test_from_bytes_binary_refused():
    with pytest.raises(BinaryFileError):
        TextFile.from_bytes(b"MZ\x00\x00\x01")


def test_from_bytes_cp1252_fallback_remembered():
    data = "café €\r\n".encode("cp1252")
    tf = TextFile.from_bytes(data)
    assert tf.encoding == "cp1252"
    assert tf.lines == ["café €\n"]
    assert tf.encoding_note is not None


def test_from_bytes_charset_hint_wins_over_cp1252():
    data = "café\n".encode("latin-1")
    tf = TextFile.from_bytes(data, charset_hint="latin1")
    assert tf.encoding == "latin1"


# ---------------------------------------------------------------------------
# re-attaching endings
# ---------------------------------------------------------------------------


def test_untouched_lines_keep_their_bytes_in_a_mixed_file():
    tf = TextFile.from_bytes(MIXED)
    new = ["a\n", "B\n", "c\n", "d\n"]  # only line 2 changed
    assert render(tf, new) == b"a\nB\r\nc\r\nd\n"


def test_inserted_block_takes_the_neighbour_above():
    tf = TextFile.from_bytes(b"a\r\nb\nc\n")
    new = ["a\n", "a2\n", "a3\n", "b\n", "c\n"]
    assert render(tf, new) == b"a\r\na2\r\na3\r\nb\nc\n"


def test_insert_at_top_takes_the_neighbour_below():
    tf = TextFile.from_bytes(b"a\r\nb\r\n")
    new = ["new\n", "a\n", "b\n"]
    assert render(tf, new) == b"new\r\na\r\nb\r\n"


def test_missing_final_newline_stays_missing():
    tf = TextFile.from_bytes(NO_FINAL)
    new = ["x\n", "Y\n", "z\n"]
    assert render(tf, new) == b"x\nY\nz"


def test_missing_final_newline_stays_missing_when_last_line_replaced():
    tf = TextFile.from_bytes(NO_FINAL)
    new = ["x\n", "y\n", "Z\n"]
    assert render(tf, new) == b"x\ny\nZ"


def test_bom_and_crlf_round_trip_one_line_edit():
    tf = TextFile.from_bytes(CRLF_BOM)
    new = ["line one\n", "LINE TWO\n", "line three\n"]
    out = render(tf, new)
    assert out == b"\xef\xbb\xbfline one\r\nLINE TWO\r\nline three\r\n"
    # Same bytes everywhere the edit did not go.
    assert out[:len(b"\xef\xbb\xbfline one\r\n")] == CRLF_BOM[:len(b"\xef\xbb\xbfline one\r\n")]


def test_identity_round_trip_is_byte_exact():
    for data in (CRLF_BOM, MIXED, NO_FINAL, b"", b"\r\n", b"only\r"):
        tf = TextFile.from_bytes(data)
        assert render(tf, list(tf.lines)) == data


def test_cp1252_written_back_as_cp1252():
    data = "café\r\n".encode("cp1252")
    tf = TextFile.from_bytes(data)
    assert render(tf, ["café\n", "niño\n"]) == "café\r\nniño\r\n".encode("cp1252")


def test_cp1252_cannot_hold_a_rocket():
    tf = TextFile.from_bytes("café\n".encode("cp1252"))
    with pytest.raises(UnencodableError) as exc:
        render(tf, ["café\n", "\U0001f680\n"])
    assert exc.value.line_no == 2
    assert exc.value.char == "\U0001f680"


def test_ascii_file_becomes_utf8_without_bom_on_unicode():
    tf = TextFile.from_bytes(b"plain\n")
    assert render(tf, ["plain\n", "\U0001f680\n"]) == "plain\n\U0001f680\n".encode()


def test_empty_file_uses_default_newline():
    tf = TextFile.empty(newline="\r\n")
    assert render(tf, ["a\n", "b\n"]) == b"a\r\nb\r\n"


def test_empty_file_without_final_newline():
    tf = TextFile.empty(newline="\n", final_newline=False)
    assert render(tf, ["a\n", "b\n"]) == b"a\nb"


def test_render_text_carries_bom_as_feff():
    tf = TextFile.from_bytes(CRLF_BOM)
    assert render_text(tf, list(tf.lines)).startswith("﻿line one\r\n")


def test_reattach_endings_all_new_lines_use_dominant():
    tf = TextFile.from_bytes(b"a\r\nb\r\n")
    assert reattach_endings(tf, ["x\n", "y\n"]) == ["x\r\n", "y\r\n"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_content_to_lines_normalizes_model_newlines():
    assert content_to_lines("a\r\nb\rc") == ["a\n", "b\n", "c\n"]
    assert content_to_lines("") == []


def test_trim_trailing_whitespace_counts():
    lines, n = trim_trailing_whitespace(["a  \n", "b\n", "c\t\n"])
    assert lines == ["a\n", "b\n", "c\n"]
    assert n == 2


def test_load_text_file(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(CRLF_BOM)
    tf = load_text_file(f)
    assert tf.bom and tf.newline == "\r\n"
