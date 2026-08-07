Use `hashed_read` when reading a file you intend to edit. It returns hashed line addresses that `hashed_replace_line`, `hashed_replace_block`, `hashed_delete_line`, and `hashed_delete_block` need.

It reads a file and prefixes every line with a `line_number|hash|` address:

```
1|0d75|from __future__ import annotations
2|0001|
3|039e|import os
4|a3f2|import sys
```

The 4-char hash is computed from the exact line content (trailing spaces included, newline excluded).
Everything after the second `|` is the line's exact content, so its leading whitespace is the line's real indentation.
Together, `(line, hash)` is a composed address: the line number locates the line, the hash confirms
the content hasn't changed. You pass these to `hashed_replace_line`, `hashed_replace_block`, `hashed_delete_line`, or `hashed_delete_block` — never guess or copy a hash by hand.

**Use `offset` (0-indexed) and `limit` to read only the section you need**.

A whole-file read (no `offset`/`limit`) of a file over the configured size threshold returns only a small head preview plus an `advisory` field. When you see the advisory, do not page through the file: search with `grep` or `find_symbol`, then read the relevant ranges with `offset`/`limit`.

**`new_content` is the line content only — never include the `11|b1c4|` address prefix.** When you replace a line, send just the code, not what `hashed_read` printed. The replace tools defend against three common slips and tell you when they act:

- If a leaked `(line_number, hash)` prefix appears in your `new_content`, it is stripped automatically. Pass `allow_literal=true` if the prefixed text is genuinely what the file should contain.
- Obvious indentation slips are corrected: a uniform comment block (every line starting with the same `///`, `//`, `#`, or `*` marker) aligns to the adjacent comment's indent; a method-chain continuation line (leading `.`) that fell to or below its statement's indent snaps back level with its `.` siblings; and a base indent sitting off the file's indent grid (e.g. 7 spaces in a 4-space file) is shifted onto it. The relative indentation between your lines is otherwise preserved exactly as you wrote it. A base that differs by full indent steps is kept and noted, assumed intentional. Pass `keep_indent=true` to write your indentation verbatim.
- If your first or last new line exactly duplicates the untouched line just outside the edited region, that duplicate is dropped. Pass `keep_duplicate=true` if you really intend the repeated line.

Every correction is reported back in the result's `content_note`, naming the affected line and the flag to override.