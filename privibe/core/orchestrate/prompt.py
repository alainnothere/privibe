"""The one user message every role call receives.

Frozen head first (task + file snapshot: byte-identical for the whole job, so
the llama.cpp prefix cache covers it), then the tail that is rewritten every
call (plan, notes, changed files, last verifier run, last worker report, the
role). Stable first, volatile last: the cache match runs up to the first byte
that changed since the previous call of any role.
"""

from __future__ import annotations

from enum import StrEnum

from privibe.core.orchestrate.ledger import TaskItem, cap_text
from privibe.core.orchestrate.verifier import VerifierResult

HEAD_TAIL_SEPARATOR = "\n\n----- state (rewritten every call) -----\n\n"


class Role(StrEnum):
    PLAN = "plan"
    BRAINSTORM = "brainstorm"
    MANAGE = "manage"
    EXECUTE = "execute"


# --- head ---------------------------------------------------------------------


def build_head(
    *, goal: str, scope: list[str], done_command: str, rendered_files: str
) -> str:
    verifier = (
        f"Done when this command exits 0: `{done_command}`"
        if done_command
        else "No verifier command. The manager decides when the job is done."
    )
    scope_text = "\n".join(f"- {s}" for s in scope) if scope else "- (none)"
    files = rendered_files if rendered_files else "(no files were pre-fed)\n"
    return (
        "# Task\n\n"
        f"{goal.strip()}\n\n"
        f"{verifier}\n\n"
        "Paths the workers may edit:\n"
        f"{scope_text}\n\n"
        "# Files (snapshot when the job started)\n\n"
        f"{files}"
    )


# --- tail ---------------------------------------------------------------------


def build_tail(
    *,
    plan: str,
    notes: str,
    tasks: list[TaskItem],
    rendered_changed: str,
    verifier: VerifierResult | None,
    last_report: str,
    report_chars: int,
    role_text: str,
) -> str:
    parts: list[str] = []
    parts.append("# Plan\n\n" + (plan if plan else "(none yet)"))
    parts.append("# Notes\n\n" + (notes if notes else "(none yet)"))
    tasks_text = "\n".join(t.render() for t in tasks) if tasks else "(none yet)"
    parts.append("# Task list\n\n" + tasks_text)
    parts.append(
        "# Changed files (current content, supersedes the snapshot)\n\n"
        + (rendered_changed if rendered_changed else "(none changed yet)")
    )
    parts.append("# Last verifier run\n\n" + _render_verifier(verifier))
    if last_report:
        parts.append("# Last worker report\n\n" + cap_text(last_report, report_chars))
    parts.append(role_text.strip())
    return "\n\n".join(parts) + "\n"


def _render_verifier(result: VerifierResult | None) -> str:
    if result is None:
        return "(not run yet)"
    verdict = "PASS" if result.passed else "FAIL"
    if result.timed_out:
        verdict = "FAIL (timed out)"
    out = result.output.strip() or "(no output)"
    return f"{verdict}, exit code {result.exit_code}\n```\n{out}\n```"


def build_message(head: str, tail: str) -> str:
    return head + HEAD_TAIL_SEPARATOR + tail


# --- roles --------------------------------------------------------------------

ROLE_PLAN = """# Your role: manager, planning

Read the task and the files. Do not write code. Answer in exactly this shape:

PLAN:
<3 to 6 sentences: what the change is, where it goes, what could go wrong>

TASKS:
1. <one concrete task a worker can finish in one call>
2. <next task>
(3 to 6 tasks, numbered, one line each)

A task is one coherent change: a function, a class, a config field with its
use. Not one line per task. The harness runs the verifier command itself after
every worker call, so never make "run the tests" a task.
"""

ROLE_BRAINSTORM = """# Your role: brainstorm

Do not write the solution and do not call tools. Think about the task and write
the notes the workers will start from, under these headings:

## Core difficulty
## Candidate approaches
## Pitfalls
## Next steps

Be concrete: name files, functions and line numbers from the snapshot. Flag
anything you are not sure of as unverified.
"""

ROLE_MANAGE = """# Your role: manager, curating

Fold the plan, the notes, the task list, the changed files and the last worker
report into one task list, then pick the single next task or declare the job
done. Do not call tools. Rules:
- Keep task ids stable; mark finished items [done]; add new sub-tasks at the end.
- If the last verifier run is FAIL the job is not done: the next task is to fix
  or change approach.
- Declare DONE only when the work is complete and, where a verifier exists, it
  passed.
- The harness runs the verifier after every worker call and shows the result
  above; never make "run the tests" a task.

Answer in exactly this shape:

TASKS:
1. [done] <text>
2. [todo] <text>
...
NEXT: <id of one todo task>   or   NEXT: DONE
"""

ROLE_EXECUTE = """# Your role: worker

Do exactly the task below with the edit tools, nothing else from the list. Make
every edit the task needs in this call; do not stop after the first. The
snapshot rows are `line|hash|content`: pass the line and hash to the hashed edit
tools; for a file under "Changed files" use those rows, not the snapshot's.
Read a file only if you need lines the message does not show.

When finished, answer in text with these two sections:

## Notes
<the complete, updated notes: replace the old notes with a curated whole that
keeps what is still true, drops what is settled, and records what you learned.
Mark anything unverified as unverified.>

## Next steps
<what remains, as the manager should see it; or "none" if the task list is done>

Your task:
{task}
"""


def role_text(role: Role, *, task: str = "") -> str:
    match role:
        case Role.PLAN:
            return ROLE_PLAN
        case Role.BRAINSTORM:
            return ROLE_BRAINSTORM
        case Role.MANAGE:
            return ROLE_MANAGE
        case Role.EXECUTE:
            return ROLE_EXECUTE.format(task=task.strip())
