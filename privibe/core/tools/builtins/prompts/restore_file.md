**Use `restore_file` instead of rewriting a file from scratch when an edit goes wrong.**

When a `write_file`, `hashed_replace_*`, `hashed_delete_*`, or `search_replace` edit produces the wrong result — or accidentally empties or clobbers a file — call `restore_file(path)` to revert the file to the content it had immediately before that edit. Do not regenerate the whole file by hand; that is slower and risks introducing new mistakes.

- Each call walks back exactly one edit. Call it again to undo the edit before that, and so on.
- If the edit you are undoing had *created* the file, restoring removes the file (it goes back to not existing) — the result will say so.
- After restoring, re-read the file (`hashed_read`) before making further edits, since its line numbers and hashes have changed.
- Restore points are kept only for edits made earlier in the current session by this agent. If none exists for the path, the tool will say so.
