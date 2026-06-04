---
name: safe-file-edit
description: "Use this skill whenever you need to edit, modify, or update the contents of any existing text file. This includes changing configuration files, source code, scripts, markdown, plain text, YAML, JSON, TOML, XML, or any other text-based file format. The skill enforces a structured workflow: state the target file, the intended change, and the objective before editing, then verify the result with a diff to confirm the change is correct and fulfills the original objective. If the change is wrong, the file is restored from backup. This prevents silent mistakes, partial edits, and changes that drift from what was actually requested. Trigger this skill for any file modification — even single-line changes — to ensure every edit is intentional, verified, and reversible."
user-invocable: true
---

# Editing Files Skill

This skill must be used whenever you need to edit or modify a text file AND YOU ARE NOT USING THE HASHED_REPLACE_LINE OR HASHED_REPLACE_BLOCK TOOLS, if you are using those tools then no need to follow the steps below.
Follow these steps exactly, in order. Do not skip any step.

## Steps

0. IF YOU ARE USING THE HASHED_REPLACE_LINE or HASHED_REPLACE_BLOCK tools then jump to 7.
1. STATE THE TARGET: Write the full path of the file you need to change as a todo entry.
2. STATE THE CHANGE: Write exactly what change needs to be made to the file as a todo entry.
3. STATE THE OBJECTIVE: Write the reason why this change is needed as a todo entry.
4. BACKUP: Copy the file to /tmp/edit-backup/ preserving the original filename with a .bak extension. Create the directory if it does not exist.
5. EDIT: Apply the change from step 2 to the original file.
6. VERIFY: Run a diff between the original file and the backup. Read the diff output and confirm that it matches the intended change from step 2 and fulfills the objective from step 3.
   - If the diff is CORRECT: Delete the backup file. Mark all todo entries as done.
   - If the diff is WRONG: Restore the original file from the backup. Report what went wrong. Do not mark the todo entries as done.
7. END
