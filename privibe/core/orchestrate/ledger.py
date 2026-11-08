from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import time
from typing import Any

TASK_FILENAME = "task.md"
PLAN_FILENAME = "plan.md"
NOTES_FILENAME = "notes.md"
TASKS_FILENAME = "tasks.json"
LOG_FILENAME = "rounds.log"
# The run's state, so a stopped run can be picked up where it was. job.json
# and snapshot.json are written once at the start; state/NNN.json once per
# clean boundary (plan written, brainstorm written, round ended); inflight.json
# while a worker runs; steer.md holds every direction the user gave.
JOB_FILENAME = "job.json"
SNAPSHOT_FILENAME = "snapshot.json"
STATE_DIRNAME = "state"
INFLIGHT_FILENAME = "inflight.json"
STEER_FILENAME = "steer.md"
# Which sub-agent session played which role call, in order: what a resumed
# conversation reads to draw the run's rows again.
CALLS_FILENAME = "calls.json"
CHECKPOINT_VERSION = 1

# Phases a checkpoint can record, in order. A resumed run skips every phase at
# or before the checkpoint's.
PHASE_PLANNED = "planned"
PHASE_BRAINSTORMED = "brainstormed"
PHASE_ROUND_END = "round_end"

# os.replace on Windows can hit a transient PermissionError while an indexer
# holds the fresh file; a short retry rides it out, the last attempt raises.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = 0.1


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so a reader never sees a truncated file: the
    bytes go to a sibling temp file first, then replace the target in one
    rename. A crash mid-write leaves the old file whole and a `.tmp` beside it.
    """
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_BACKOFF_S * attempt)


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


@dataclass(frozen=True)
class Checkpoint:
    """The run at one clean boundary, complete enough to continue from: the
    counters, the stall keys, the last verifier run and worker report, the
    task list, the notes, and the listed files as they were. The plan is not
    here because it never changes after the plan phase; the job and the file
    snapshot are not here because they never change at all (job.json,
    snapshot.json).
    """

    number: int
    phase: str
    rounds: int
    calls: int
    verifier: dict[str, Any] | None
    last_report: str
    last_task_key: str | None
    done_keys: list[str]
    tasks: list[TaskItem]
    notes: str
    # Listed file -> content at the boundary; None when the file was missing.
    files: dict[str, str | None] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "number": self.number,
            "phase": self.phase,
            "rounds": self.rounds,
            "calls": self.calls,
            "verifier": self.verifier,
            "last_report": self.last_report,
            "last_task_key": self.last_task_key,
            "done_keys": list(self.done_keys),
            "tasks": [{"id": t.id, "text": t.text, "done": t.done} for t in self.tasks],
            "notes": self.notes,
            "files": self.files,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Checkpoint:
        return cls(
            number=int(raw["number"]),
            phase=str(raw["phase"]),
            rounds=int(raw["rounds"]),
            calls=int(raw["calls"]),
            verifier=raw.get("verifier"),
            last_report=str(raw.get("last_report", "")),
            last_task_key=raw.get("last_task_key"),
            done_keys=[str(k) for k in raw.get("done_keys", [])],
            tasks=[
                TaskItem(id=int(t["id"]), text=t["text"], done=bool(t["done"]))
                for t in raw.get("tasks", [])
            ],
            notes=str(raw.get("notes", "")),
            files=dict(raw.get("files", {})),
        )


class LedgerStateError(ValueError):
    """The ledger cannot be resumed as asked: no checkpoints, a missing
    checkpoint number, or no job.json.
    """


_CHECKPOINT_NAME = re.compile(r"^(\d{3,})\.json$")


class Ledger:  # noqa: PLR0904  every write of the run goes through here
    """The files the roles share. Every write goes through here so the caps
    hold and `rounds.log` records who wrote what, when. Every write is atomic,
    so a run that dies mid-write leaves whole files behind.
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

    @property
    def job_path(self) -> Path:
        return self.root / JOB_FILENAME

    @property
    def snapshot_path(self) -> Path:
        return self.root / SNAPSHOT_FILENAME

    @property
    def state_dir(self) -> Path:
        return self.root / STATE_DIRNAME

    @property
    def inflight_path(self) -> Path:
        return self.root / INFLIGHT_FILENAME

    @property
    def steer_path(self) -> Path:
        return self.root / STEER_FILENAME

    @property
    def calls_path(self) -> Path:
        return self.root / CALLS_FILENAME

    # --- task / plan / notes ----------------------------------------------
    def write_task(self, text: str) -> None:
        write_atomic(self.task_path, text.strip() + "\n")

    def read_plan(self) -> str:
        return self._read(self.plan_path)

    def write_plan(self, text: str) -> str:
        capped = cap_text(text, self.plan_chars)
        write_atomic(self.plan_path, capped + "\n")
        return capped

    def read_notes(self) -> str:
        return self._read(self.notes_path)

    def write_notes(self, text: str) -> str:
        capped = cap_text(text, self.notes_chars)
        write_atomic(self.notes_path, capped + "\n")
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
        write_atomic(self.tasks_path, _dumps(payload))

    # --- run state ----------------------------------------------------------
    def write_job(self, job: dict[str, Any]) -> None:
        """The job as the run saw it (goal, files, scope, done_command, the
        effective max_rounds, cwd), written once so a resume needs nothing
        from the caller but the ledger directory.
        """
        write_atomic(self.job_path, _dumps(job))

    def read_job(self) -> dict[str, Any] | None:
        if not self.job_path.is_file():
            return None
        return json.loads(self.job_path.read_text(encoding="utf-8"))

    def write_snapshot(self, paths: list[str], contents: dict[str, str]) -> None:
        """The pre-fed files as they were when the job started: the frozen
        head is built from these, so a resumed run renders the same bytes.
        """
        write_atomic(
            self.snapshot_path, _dumps({"paths": list(paths), "contents": contents})
        )

    def read_snapshot(self) -> tuple[list[str], dict[str, str]] | None:
        if not self.snapshot_path.is_file():
            return None
        raw = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        return [str(p) for p in raw["paths"]], dict(raw["contents"])

    def checkpoints(self) -> list[Checkpoint]:
        """Every checkpoint on disk, in number order."""
        if not self.state_dir.is_dir():
            return []
        found: list[Checkpoint] = []
        for path in self.state_dir.iterdir():
            if _CHECKPOINT_NAME.match(path.name):
                raw = json.loads(path.read_text(encoding="utf-8"))
                found.append(Checkpoint.from_json(raw))
        return sorted(found, key=lambda c: c.number)

    def latest_checkpoint(self) -> Checkpoint | None:
        found = self.checkpoints()
        return found[-1] if found else None

    def checkpoint(self, number: int) -> Checkpoint:
        for cp in self.checkpoints():
            if cp.number == number:
                return cp
        raise LedgerStateError(f"no checkpoint {number} in {self.root}")

    def next_checkpoint_number(self) -> int:
        latest = self.latest_checkpoint()
        return 1 if latest is None else latest.number + 1

    def write_checkpoint(self, cp: Checkpoint) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / f"{cp.number:03d}.json"
        write_atomic(path, _dumps(cp.to_json()))
        return path

    def drop_checkpoints_after(self, number: int) -> list[int]:
        """Delete every checkpoint above `number`: continuing from an earlier
        boundary abandons the later ones, the way a rewind does. Returns the
        numbers dropped.
        """
        dropped: list[int] = []
        for cp in self.checkpoints():
            if cp.number > number:
                (self.state_dir / f"{cp.number:03d}.json").unlink()
                dropped.append(cp.number)
        return dropped

    def write_inflight(self, *, round_no: int, task_id: int, text: str) -> None:
        """A worker is running this task. Cleared when the round ends, so a
        file left behind means the run stopped mid-worker.
        """
        write_atomic(
            self.inflight_path, _dumps({"round": round_no, "id": task_id, "text": text})
        )

    def read_inflight(self) -> dict[str, Any] | None:
        if not self.inflight_path.is_file():
            return None
        return json.loads(self.inflight_path.read_text(encoding="utf-8"))

    def clear_inflight(self) -> None:
        self.inflight_path.unlink(missing_ok=True)

    def append_steer(self, text: str) -> None:
        """Add a direction from the user. Kept whole and in order: every later
        role call sees all of them.
        """
        text = text.strip()
        if not text:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M")
        current = self.read_steer()
        entry = f"({stamp}) {text}"
        write_atomic(
            self.steer_path, (current + "\n\n" + entry if current else entry) + "\n"
        )

    def read_steer(self) -> str:
        return self._read(self.steer_path)

    def record_call(self, label: str, session_dir: str, *, tag: str = "") -> None:
        """Append one role call: its section label, the worker session that
        played it ("" when the worker logged nowhere) and the short tag its
        rows carry.
        """
        calls = self.read_calls()
        calls.append({"label": label, "session_dir": session_dir, "tag": tag})
        write_atomic(self.calls_path, _dumps(calls))

    def read_calls(self) -> list[dict[str, str]]:
        if not self.calls_path.is_file():
            return []
        raw = json.loads(self.calls_path.read_text(encoding="utf-8"))
        return [
            {
                "label": str(c.get("label", "")),
                "session_dir": str(c.get("session_dir", "")),
                "tag": str(c.get("tag", "")),
            }
            for c in raw
        ]

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
        write_atomic(path, text)
        return path

    @staticmethod
    def _read(path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
