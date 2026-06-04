**Prefer `find_symbol` over `grep` when looking for a named symbol** (function, method, class, interface). `find_symbol` returns the definition body with hashed lines in one call — grep only returns matching lines and still requires a follow-up read. Only fall back to `grep` when `find_symbol` is unavailable or the search does not target a specific named symbol (e.g. searching for a string literal, an error message, or a usage pattern across files).

Use `grep` to recursively search for a regular expression pattern in files.

- It's very fast and automatically ignores files that you should not read like .pyc files, .venv directories, etc.
- Use this to find where functions are defined, how variables are used, or to locate specific error messages.
