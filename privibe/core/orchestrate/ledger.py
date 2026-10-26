from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

TASK_FILENAME = "task.md"
PLAN_FILENAME = "plan.md"
NOTES_FILENAME = "notes.md"
TASKS_FILENAME = "tasks.json"
LOG_FILENAME = "rounds.log"

TRUNCATION_MARK = "\n[... cut to {limit} characters by the harness ...]"


@dataclass(frozen=True)
class TaskItem:
    id: int
    text: str
    done: bool = False

    def render(self) -> str:
        mark = "done" if self.done else "todo"
        return f"{self.id}. [{mark}] {self.text}"


def cap_text(text: str, limit: int) -> str:
    """Cut `text` to `limit` characters, marker included, so a capped file is
    never silently a prefix: the reader can see it was cut and by whom.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    mark = TRUNCATION_MARK.format(limit=limit)
    keep = max(0, limit - len(mark))
    return text[:keep].rstrip() + mark


class Ledger:
    """The files the roles share. Every write goes through here so the caps
    hold and `rounds.log` records who wrote what, when.
    """

    def __init__(self, root: Path, *, plan_chars: int, notes_chars: int) -> None:
        self.root = root
        self.plan_chars = plan_chars
        self.notes_chars = notes_chars
        self.root.mkdir(parents=True, exist_ok=True)

    # --- paths -----------------------------------------------------------
    @property
    def task_path(self) -> Path:
        return self.root / TASK_FILENAME

    @property
    def plan_path(self) -> Path:
        return self.root / PLAN_FILENAME

    @property
    def notes_path(self) -> Path:
        return self.root / NOTES_FILENAME

    @property
    def tasks_path(self) -> Path:
        return self.root / TASKS_FILENAME

    @property
    def log_path(self) -> Path:
        return self.root / LOG_FILENAME

    # --- task / plan / notes ----------------------------------------------
    def write_task(self, text: str) -> None:
        self.task_path.write_text(text.strip() + "\n", encoding="utf-8")

    def read_plan(self) -> str:
        return self._read(self.plan_path)

    def write_plan(self, text: str) -> str:
        capped = cap_text(text, self.plan_chars)
        self.plan_path.write_text(capped + "\n", encoding="utf-8")
        return capped

    def read_notes(self) -> str:
        return self._read(self.notes_path)

    def write_notes(self, text: str) -> str:
        capped = cap_text(text, self.notes_chars)
        self.notes_path.write_text(capped + "\n", encoding="utf-8")
        return capped

    # --- task list ---------------------------------------------------------
    def read_tasks(self) -> list[TaskItem]:
        if not self.tasks_path.is_file():
            return []
        raw = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        return [
            TaskItem(id=int(t["id"]), text=t["text"], done=bool(t["done"])) for t in raw
        ]

    def write_tasks(self, tasks: list[TaskItem]) -> None:
        payload = [{"id": t.id, "text": t.text, "done": t.done} for t in tasks]
        self.tasks_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # --- log ---------------------------------------------------------------
    def log(self, event: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} {event}\n")

    def save_transcript(self, name: str, text: str) -> Path:
        """Keep a role call's full answer beside the ledger, for the human.
        Never fed back to the model; the capped files are.
        """
        path = self.root / "calls" / f"{name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def _read(path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
