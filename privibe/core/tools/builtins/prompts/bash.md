Use the `bash` tool to run one-off shell commands.

**Key characteristics:**
- **Stateless**: Each command runs independently in a fresh environment

**Timeout:**
- The `timeout` argument controls how long the command can run before being killed
- When `timeout` is not specified (or set to `None`), the config default is used
- If a command is timing out, do not hesitate to increase the timeout using the `timeout` argument

**IMPORTANT: Use dedicated tools if available instead of these bash commands:**
**IMPORTANT: DO NOT DO REPLACEMENTS USING SED OR ANY SIMILAR TOOL, IT'S A SHOTGUN TO THE FACE, ANYTHING CHANGED UNINTENTIONALLY WILL MEAN COSTLY REWORK LATER... DO NOT DO SMARTASS REPLACEMENTS, DO SMALL SET OF CHANGES WHERE THE CHANGE IS VERY INTENTIONAL**

**File Operations - DO NOT USE:**
- `cat filename` → Use `hashed_read(path="filename")`
- `head -n 20 filename` → Use `hashed_read(path="filename", limit=20)`
- `tail -n 20 filename` → Read with offset: `hashed_read(path="filename", offset=<line_number>, limit=20)`
- `sed -n '100,200p' filename` → Use `hashed_read(path="filename", offset=99, limit=101)`
- `less`, `more`, `vim`, `nano` → Use `hashed_read` with offset/limit for navigation
- `echo "content" > file` → Use `write_file(path="file", content="content")`
- `echo "content" >> file` → Read first, then `write_file` with overwrite=true

**Search Operations - DO NOT USE:**
- `grep -r "pattern" .` → Use `grep(pattern="pattern", path=".")`
- `find . -name "*.py"` → Use `bash("ls -la")` for current dir or `grep` with appropriate pattern
- `ag`, `ack`, `rg` commands → Use the `grep` tool
- `locate` → Use `grep` tool

**File Modification - DO NOT USE:**
- `sed -i 's/old/new/g' file` → Use `hashed_replace_line` and `hashed_replace_block` tools
- `awk` for file editing → Use `hashed_replace_line` and `hashed_replace_block` tools
- Any in-place file editing → Use `hashed_replace_line` and `hashed_replace_block` tools

**APPROPRIATE bash uses:**
- System information: `pwd`, `whoami`, `date`, `uname -a`
- Directory listings: `ls -la`, `tree` (if available)
- Git operations: `git status`, `git log --oneline -10`, `git diff`
- Process info: `ps aux | grep process`, `top -n 1`
- Network checks: `ping -c 1 google.com`, `curl -I https://example.com`
- Package management: `pip list`, `npm list`
- Environment checks: `env | grep VAR`, `which python`
- File metadata: `stat filename`, `file filename`, `wc -l filename`

**Example: Reading a large file efficiently**

WRONG:
```bash
bash("cat large_file.txt")  # May hit size limits
bash("head -1000 large_file.txt")  # Inefficient
```

RIGHT:
```python
# First chunk
hashed_read(path="large_file.txt", limit=1000)
# If was_truncated=true, read next chunk
hashed_read(path="large_file.txt", offset=1000, limit=1000)
```

**Example: Searching for patterns**

WRONG:
```bash
bash("grep -r 'TODO' src/")  # Don't use bash for grep
bash("find . -type f -name '*.py' | xargs grep 'import'")  # Too complex
```

RIGHT:
```python
grep(pattern="TODO", path="src/")
grep(pattern="import", path=".")
```

**Remember:** Bash is best for quick system checks and git operations. For file operations, searching, and editing, always use the dedicated tools when they are available.
