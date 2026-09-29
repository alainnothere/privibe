"""ASCII transliteration: small table, NFKD for accents, everything else
dropped and named.
"""

from __future__ import annotations

import pytest

from privibe.core.utils.asciify import (
    asciify,
    format_ascii_notes,
    resolve_ascii,
    session_ascii_override,
    set_session_ascii_override,
)


@pytest.fixture(autouse=True)
def _clear_session_override():
    set_session_ascii_override(None)
    yield
    set_session_ascii_override(None)


def test_pure_ascii_untouched():
    r = asciify("plain text\nline two")
    assert r.text == "plain text\nline two"
    assert not r.changed


def test_typographic_set():
    r = asciify("a — b – c … → “q” ‘s’ x")
    assert r.text == "a - b - c ... -> \"q\" 's' x"
    assert r.changed


def test_accents_via_nfkd():
    assert asciify("café señal über").text == "cafe senal uber"


def test_table_symbols():
    assert asciify("12€ at 20°").text == "12EUR at 20 deg"
    assert asciify("a ≤ b ≠ c").text == "a <= b != c"


def test_emoji_dropped_and_named():
    r = asciify("done ✅ \U0001f680")
    assert r.text == "done [x] "
    dropped = [x for x in r.replacements if x.replacement == ""]
    assert [x.char for x in dropped] == ["\U0001f680"]


def test_variation_selector_silently_dropped():
    r = asciify("✓️")
    assert r.text == "[x]"
    # The selector is noise; only the check mark is reported.
    assert [x.char for x in r.replacements] == ["✓"]


def test_line_numbers_are_right():
    r = asciify("ok\nseñal\nfine\n€")
    assert [(x.line_no, x.char) for x in r.replacements] == [(2, "ñ"), (4, "€")]


def test_notes_name_line_and_char_and_flag():
    r = asciify("x\nseñal — ok")
    notes = format_ascii_notes(r.replacements)
    assert notes[0].startswith("line 2: non-ASCII written as ASCII: ")
    assert "latin small letter n with tilde -> 'n'" in notes[0]
    assert "em dash -> '-'" in notes[0]
    assert notes[-1] == "(pass allow_unicode=true to write these characters verbatim as UTF-8)"


def test_notes_flag_quote_risk():
    r = asciify("it’s")
    notes = format_ascii_notes(r.replacements)
    assert "(check quoting)" in notes[0]


def test_notes_are_capped():
    text = "\n".join("é" for _ in range(30))
    r = asciify(text)
    notes = format_ascii_notes(r.replacements, cap=5)
    assert len(notes) == 5 + 2
    assert notes[5] == "...and 25 more lines with non-ASCII replacements"


def test_repeated_char_on_one_line_counted_once_with_multiplier():
    r = asciify("———")
    notes = format_ascii_notes(r.replacements)
    assert "em dash -> '-' x3" in notes[0]


def test_resolve_ascii_precedence():
    # explicit flag wins over everything
    assert resolve_ascii(True, True) is False
    assert resolve_ascii(False, False) is True
    # config default when nothing else
    assert resolve_ascii(None, True) is True
    assert resolve_ascii(None, False) is False
    # session override beats config
    set_session_ascii_override(False)
    assert session_ascii_override() is False
    assert resolve_ascii(None, True) is False
    assert resolve_ascii(False, True) is True
