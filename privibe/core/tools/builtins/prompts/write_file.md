Use `write_file` to create a new file, or to overwrite an existing one by setting `overwrite: true` — without it, writing to an existing path fails, so you cannot clobber a file by accident. Parent directories are created automatically.

**Best practices:**

- ALWAYS read a file (`hashed_read`) before overwriting it.
- Prefer `hashed_replace_line`/`hashed_replace_block` for editing existing files — overwriting rewrites the whole file and risks losing content you did not intend to touch.
- Never write new files unless the task requires it; prefer modifying existing ones.
- Never proactively create documentation files (*.md) or READMEs unless explicitly requested.
- Line endings, encoding and BOM are handled by the tool: a new file follows the repo's conventions (.gitattributes, .editorconfig, sibling files), an overwrite keeps the file's own. Write content with plain newlines and do not inspect or correct any of this yourself. Set `newline` or `encoding` only when you know something the repo does not.
