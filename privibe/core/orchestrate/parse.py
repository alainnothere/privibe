"""Parsers for the text the roles answer with.

The formats are flat, labelled sections rather than JSON on purpose: small
local models get a numbered list right far more often than nested JSON, and a
miss here costs a whole round.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from privibe.core.orchestrate.ledger import TaskItem


class ParseError(ValueError):
    pass


_TASK_LINE = re.compile(
    r"^\s*(?P<id>\d+)[.)]\s*(?:\[(?P<mark>[^\]]*)\]\s*)?(?P<text>.+?)\s*$"
)
_NEXT_LINE = re.compile(
    r"^\s*NEXT\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_SECTION = re.compile(r"^\s*(PLAN|TASKS)\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_DONE_WORDS = {"done", "x", "complete", "completed", "finished"}


def _strip_fences(text: str) -> str:
    # Models wrap the whole answer in ``` now and then; the shape inside is
    # what matters.
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        return inner.rsplit("```", 1)[0]
    return text


def _parse_task_lines(block: str) -> list[TaskItem]:
    tasks: list[TaskItem] = []
    seen: set[int] = set()
    for line in block.splitlines():
        m = _TASK_LINE.match(line)
        if not m:
            continue
        task_id = int(m.group("id"))
        if task_id in seen:
            continue
        seen.add(task_id)
        mark = (m.group("mark") or "").strip().lower()
        tasks.append(
            TaskItem(id=task_id, text=m.group("text").strip(), done=mark in _DONE_WORDS)
        )
    return tasks


@dataclass(frozen=True)
class PlanAnswer:
    plan: str
    tasks: list[TaskItem]


def parse_plan(text: str) -> PlanAnswer:
    text = _strip_fences(text)
    sections = {m.group(1).upper(): m.end() for m in _SECTION.finditer(text)}
    if "TASKS" not in sections:
        raise ParseError("no 'TASKS:' section")
    tasks_start = sections["TASKS"]
    if "PLAN" in sections and sections["PLAN"] < tasks_start:
        plan = text[sections["PLAN"] : text.rfind("TASKS", 0, tasks_start)].strip()
    else:
        plan = text[: text.rfind("TASKS", 0, tasks_start)].strip()
    tasks = _parse_task_lines(text[tasks_start:])
    if not tasks:
        raise ParseError("'TASKS:' section has no numbered tasks")
    if not plan:
        raise ParseError("no plan text before 'TASKS:'")
    return PlanAnswer(plan=plan, tasks=tasks)


@dataclass(frozen=True)
class ManageAnswer:
    tasks: list[TaskItem]
    next_id: int | None  # None means DONE
    done: bool

    def next_task(self) -> TaskItem | None:
        if self.next_id is None:
            return None
        for t in self.tasks:
            if t.id == self.next_id:
                return t
        return None


def parse_manage(text: str) -> ManageAnswer:
    text = _strip_fences(text)
    nexts = list(_NEXT_LINE.finditer(text))
    if not nexts:
        raise ParseError("no 'NEXT:' line")
    value = nexts[-1].group("value").strip().strip("`*").strip()
    tasks_text = text[: nexts[-1].start()]
    tasks = _parse_task_lines(tasks_text)
    if value.lower().rstrip(".") == "done":
        return ManageAnswer(tasks=tasks, next_id=None, done=True)
    m = re.match(r"^(?:task\s*)?#?(\d+)", value, re.IGNORECASE)
    if not m:
        raise ParseError(f"'NEXT:' must be a task id or DONE, got {value!r}")
    next_id = int(m.group(1))
    if not tasks:
        raise ParseError("no numbered task list before 'NEXT:'")
    chosen = next((t for t in tasks if t.id == next_id), None)
    if chosen is None:
        raise ParseError(f"'NEXT: {next_id}' names no task in the list")
    if chosen.done:
        raise ParseError(f"'NEXT: {next_id}' names a task marked done")
    return ManageAnswer(tasks=tasks, next_id=next_id, done=False)


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(?P<title>[^\n]+?)\s*#*\s*$", re.MULTILINE)


def extract_section(text: str, title: str) -> str | None:
    """Body of the markdown section whose heading is `title` (case-insensitive),
    up to the next heading of any level. None when absent.
    """
    headings = list(_HEADING.finditer(text))
    for i, h in enumerate(headings):
        if h.group("title").strip().lower() == title.lower():
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            return text[h.end() : end].strip()
    return None


def parse_worker_notes(text: str) -> str | None:
    """The worker's replacement notes, or None when it gave none (the ledger
    then keeps the previous notes).
    """
    body = extract_section(text, "Notes")
    if body is None:
        return None
    return body or None
