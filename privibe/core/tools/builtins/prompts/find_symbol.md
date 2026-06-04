Use `find_symbol` as the **first choice** when looking for any named symbol (function, method, class, interface, trait). Prefer it over `grep` — grep only returns matching lines and requires a follow-up file read, while `find_symbol` returns the full definition body with hashed lines in one call. Fall back to `grep` only when `find_symbol` cannot locate the symbol or the search targets something other than a named definition.

**Why this exists:** The grep → read-file → scroll workflow costs 3+ round trips and returns far more than you need. `find_symbol` collapses this into one call.

**Parameters:**

- `symbol` — name or regex pattern: `DoSomething`, `I.*Repository`, `get_\w+`. Passed directly to ripgrep.
- `path` — directory or glob: `src/`, `**/*.cs`, `lib/foo.py`. Defaults to `.`
- `kind` — `"function"` (includes methods), `"class"` (includes structs/records/enums), `"interface"` (includes traits). Restricts search to definition lines. Omit for broad search.
- `extensions` — explicit list like `["cs", "ts"]`. Inferred from path glob when possible.

**Supported languages:** Python (`.py`), C# (`.cs`), Java (`.java`), TypeScript/JS (`.ts`, `.tsx`, `.js`, `.jsx`), Go (`.go`), Rust (`.rs`)

**Output format:**
```
=== src/Services/FooService.cs:42 ===
   37 a3f2  // 5 lines of context before
   ...
   42 b891  public class FooService : IFooService
   43 c4d1  {
   ...
   91 d0e2  }
```

Each block starts 5 lines before the definition and includes the full body (up to 50 lines), all with hashed line numbers usable directly in `hashed_replace_line` or `hashed_replace_block`.

**Examples:**

Find all C# classes named UserService:
```json
{"symbol": "UserService", "path": "src/", "kind": "class", "extensions": ["cs"]}
```

Find the interface and all its implementations:
```json
{"symbol": "IOrderRepository", "path": ".", "kind": "interface"}
```

Find all C# interfaces matching a pattern:
```json
{"symbol": "I.*Service", "path": "src/", "kind": "interface", "extensions": ["cs"]}
```

**Notes:**
- Returns up to 8 matches. If more exist, narrow with `extensions` or a tighter `symbol` pattern.
- Body truncated at 50 lines — use `hashed_read` with `offset`/`limit` for the rest.
- When `kind` is omitted, call sites may appear alongside definitions.
- Hashes in the output are valid for `hashed_replace_line` and `hashed_replace_block` — no re-read needed for targeted edits.
- Body extraction is best-effort: brace counting for C-family, indentation for Python. Comments or strings containing braces may occasionally confuse the boundary detection.
