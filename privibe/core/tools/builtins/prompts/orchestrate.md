Use `orchestrate` to hand a whole coding job to a loop of fresh model calls: a manager writes a plan and keeps a task list, workers do one task each, and a verifier command decides when it is done. Every call starts from an empty context and reads state only from files the loop keeps on disk.

## When to use it

- The job has several steps and a command that can tell when it is done (a test, a build, a script).
- Doing it inline would mean many rounds of reading, editing and re-running in this conversation.

Not for questions, not for one-edit fixes, and not for work that needs the user mid-way: the loop cannot ask them anything.

## How to call it

- `goal`: the complete request. The workers see nothing from this conversation; write what they need.
- `files`: the files every call is shown, whole. Put read-only context first and the files you expect to be edited last.
- `scope`: the directories the workers may edit. Anything else is refused.
- `done_command`: a shell command that exits 0 when the job is done. Give one whenever you can; without it the result is `done_unverified` and means only that the manager stopped.

The result reports the status, the last verifier run, the files that changed and the ledger directory holding `plan.md`, `notes.md`, `tasks.json` and every call's answer. Check the changed files yourself before telling the user the job is done.
