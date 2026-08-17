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
- `tail -n 20 filename` → Read from a line: `hashed_read(path="filename", start_line=<line_number>, limit=20)`
- `sed -n '100,200p' filename` → Use `hashed_read(path="filename", start_line=100, limit=101)`
- `less`, `more`, `vim`, `nano` → Use `hashed_read` with start_line/limit for navigation
- `echo "content" > file` → Use `write_file(path="file", content="content")`
- `echo "content" >> file` → Read first, then `write_file` with overwrite=true

**Search Operations - DO NOT USE:**
- `grep -r "pattern" .`, `ag`, `ack`, `rg` → Use the `grep` tool: it is fast and automatically skips files you should not read

**File Modification - DO NOT USE:**
- `sed -i 's/old/new/g' file`, `awk` editing, or any in-place file editing → Use `hashed_replace_line` and `hashed_replace_block`

**APPROPRIATE bash uses:**
- Git operations: `git status`, `git log --oneline -10`, `git diff`
- Directory listings and finding files by name: `ls -la`, `find . -name "*.py"`
- System and environment checks: `pwd`, `uname -a`, `env | grep VAR`, `which python`
- File metadata: `stat filename`, `wc -l filename`
- Package management: `pip list`, `npm list`

**Remember:** Bash is best for quick system checks, finding files by name, and git operations. For reading, content search, and editing files, always use the dedicated tools when they are available.
