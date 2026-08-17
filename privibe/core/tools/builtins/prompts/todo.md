Use the `todo` tool to manage a task list that tracks your progress through multi-step work.

## How it works

- `action: "read"` returns the current list.
- `action: "write"` replaces the ENTIRE list with the `todos` you send — any todo you omit is removed, so always send the complete list you want to keep.

## When to use

- Complex multi-step tasks (3+ distinct steps)
- Multiple tasks provided by the user (numbered or comma-separated)
- After receiving new instructions — capture the requirements as todos
- Tracking progress on ongoing work

Skip it for single straightforward tasks, trivial operations, and purely conversational requests — a todo list there is noise, not organization.

## Discipline

- Only ONE task `in_progress` at a time; mark it BEFORE starting the work.
- Mark `completed` IMMEDIATELY when a task is fully done. Never mark complete while tests fail, the implementation is partial, or errors are unresolved.
- When blocked, keep the task `in_progress` and add a new task describing what must be resolved.
- Write specific, actionable items; break big tasks into manageable steps.
- Remove irrelevant tasks from the list entirely rather than marking them `cancelled`.
