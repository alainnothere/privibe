**Prefer `hashed_read` over `read_file`** when you intend to edit the file afterward. `hashed_read` returns line numbers and hashes that `hashed_replace_line` and `hashed_replace_block` need — reading with `read_file` first means you'll need a second read anyway. Only use `read_file` when you need to understand file content with no intent to edit, or when `hashed_read` is unavailable.

Use `read_file` to read the content of a file. It's designed to handle large files safely.

- By default, it reads from the beginning of the file.
- Use `start_line` (1-based line number) and `limit` (number of lines) to read specific parts or chunks of a file. This is efficient for exploring large files.
- The result includes `was_truncated: true` if the file content was cut short due to size limits.
- A whole-file read (no `start_line`/`limit`) of a file over the configured size threshold returns only a small head preview plus an `advisory` field. When you see the advisory, do not page through the file: search with `grep` or `find_symbol`, then read the relevant ranges. Do NOT fall back to `bash` `cat` on that file either — the advisory means the file is too big to dump into context by any route.
- This is more efficient than using `bash` with `cat` or `wc`.

**Strategy for large files:**

1. Call `read_file` with a `limit` (e.g., 1000 lines) to get the start of the file.
2. If `was_truncated` is true, the file is large. STOP and assess: do you already have enough information to answer the user's question? If yes, respond immediately — do not keep reading.
3. If you need more, prefer targeted reads (e.g., jump to a specific line, read the last 100 lines, search for a relevant section) over reading sequentially chunk by chunk.
4. Do not call `read_file` more than 3 times on the same file without responding to the user first.

**Do not read or explore:**
- Model checkpoint directories or weight files (.bin, .safetensors, .pt, .gguf, optimizer states, etc.)
- Binary files of any kind
- Entire directory trees of training runs or large codebases. If the user provides paths to such files, treat them as references. Do not open them unless the user explicitly asks you to inspect a specific file.
