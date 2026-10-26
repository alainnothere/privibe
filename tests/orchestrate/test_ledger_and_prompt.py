"""The head of the role message is the KV-cache prefix for the whole job: it
must be byte-identical across rounds no matter what the roles write, and the
ledger caps must hold on every write. Checked at the byte level, on the string
the loop hands the model, not on components.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from privibe.core.orchestrate.ledger import TRUNCATION_MARK, Ledger, TaskItem, cap_text
from privibe.core.orchestrate.prompt import (
    HEAD_TAIL_SEPARATOR,
    Role,
    build_head,
    build_message,
    build_tail,
    role_text,
)
from privibe.core.orchestrate.snapshot import Bundle, BundleTooLargeError, render_file
from privibe.core.orchestrate.verifier import VerifierResult
from privibe.core.tools.builtins.hashed_read import _line_hash


class TestCapText:
    def test_under_limit_untouched(self) -> None:
        assert cap_text("  short  ", 100) == "short"

    def test_over_limit_is_exactly_limit_with_marker(self) -> None:
        capped = cap_text("x" * 500, 120)
        assert len(capped) == 120
        assert capped.endswith(TRUNCATION_MARK.format(limit=120))


class TestLedger:
    def test_caps_hold_on_write(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path, plan_chars=600, notes_chars=600)
        ledger.write_plan("p" * 5000)
        ledger.write_notes("n" * 5000)
        assert len(ledger.read_plan()) == 600
        assert len(ledger.read_notes()) == 600

    def test_tasks_round_trip(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path, plan_chars=600, notes_chars=600)
        tasks = [TaskItem(1, "a", True), TaskItem(2, "b")]
        ledger.write_tasks(tasks)
        assert ledger.read_tasks() == tasks

    def test_transcripts_go_beside_the_ledger(self, tmp_path: Path) -> None:
        ledger = Ledger(tmp_path, plan_chars=600, notes_chars=600)
        path = ledger.save_transcript("001-plan", "answer")
        assert path == tmp_path / "calls" / "001-plan.md"
        assert path.read_text() == "answer"


class TestBundle:
    def test_render_uses_hashed_read_format(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1\ny = 2\n")
        rendered = render_file(str(f), f.read_text())
        assert rendered.splitlines()[0] == f"### {f} (2 lines)"
        assert rendered.splitlines()[1] == f"1|{_line_hash('x = 1')}|x = 1"

    def test_byte_cap_is_measured_not_estimated(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("line\n" * 100)
        rendered = render_file(str(f), f.read_text())
        limit = len(rendered.encode()) - 1
        with pytest.raises(BundleTooLargeError) as exc:
            Bundle.load([str(f)], max_bytes=limit)
        assert exc.value.measured == len(rendered.encode())
        Bundle.load([str(f)], max_bytes=limit + 1)

    def test_changed_files_and_deletion(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("same\n")
        b.write_text("before\n")
        bundle = Bundle.load([str(a), str(b)], max_bytes=10_000)
        assert bundle.changed_files() == []
        b.write_text("after\n")
        assert bundle.changed_files() == [str(b)]
        assert "after" in bundle.render_changed()
        b.unlink()
        assert bundle.changed_files() == [str(b)]
        assert "(deleted)" in bundle.render_changed()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestMessageLayout:
    def test_head_is_byte_stable_across_rounds(self, tmp_path: Path) -> None:
        target = tmp_path / "t.py"
        target.write_text("def f():\n    return 1\n")
        bundle = Bundle.load([str(target)], max_bytes=10_000)
        head = build_head(
            goal="make f return 2",
            scope=[str(tmp_path)],
            done_command="pytest",
            rendered_files=bundle.render_frozen(),
        )
        fingerprint = _sha(head)

        # Round 1: nothing written yet.
        m1 = build_message(
            head,
            build_tail(
                plan="",
                notes="",
                tasks=[],
                rendered_changed=bundle.render_changed(),
                verifier=None,
                last_report="",
                report_chars=100,
                role_text=role_text(Role.PLAN),
            ),
        )
        # Round 2: the worker edited the target, the ledger has content, the
        # verifier ran. Only the tail may differ.
        target.write_text("def f():\n    return 2\n")
        m2 = build_message(
            head,
            build_tail(
                plan="the plan",
                notes="the notes",
                tasks=[TaskItem(1, "edit", True)],
                rendered_changed=bundle.render_changed(),
                verifier=VerifierResult(exit_code=1, output="boom"),
                last_report="I edited it",
                report_chars=100,
                role_text=role_text(Role.MANAGE),
            ),
        )
        for m in (m1, m2):
            head_part, _sep, _tail = m.partition(HEAD_TAIL_SEPARATOR)
            assert _sha(head_part) == fingerprint
        assert m1 != m2
        # The frozen head still shows the snapshot, the tail the current file.
        assert "return 1" in m2.partition(HEAD_TAIL_SEPARATOR)[0]
        assert "return 2" in m2.partition(HEAD_TAIL_SEPARATOR)[2]

    def test_role_text_is_last(self) -> None:
        tail = build_tail(
            plan="p",
            notes="n",
            tasks=[],
            rendered_changed="",
            verifier=None,
            last_report="",
            report_chars=100,
            role_text=role_text(Role.EXECUTE, task="1. [todo] do it"),
        )
        assert tail.rstrip().endswith("1. [todo] do it")

    def test_last_report_only_reaches_the_manager(self) -> None:
        kwargs = dict(
            plan="p",
            notes="n",
            tasks=[],
            rendered_changed="",
            verifier=None,
            report_chars=100,
        )
        with_report = build_tail(
            last_report="REPORT", role_text=role_text(Role.MANAGE), **kwargs
        )
        without = build_tail(
            last_report="", role_text=role_text(Role.EXECUTE), **kwargs
        )
        assert "REPORT" in with_report
        assert "Last worker report" not in without


class TestChangedFileHunks:
    def test_only_changed_regions_with_absolute_numbers(self) -> None:
        from privibe.core.orchestrate.snapshot import (
            CHANGED_CONTEXT_LINES,
            render_changed_file,
        )

        old = "".join(f"line {i}\n" for i in range(1, 201))
        new = old.replace("line 50\n", "line 50 changed\n").replace(
            "line 180\n", "line 180 changed\nline 180b\n"
        )
        out = render_changed_file("f.txt", old, new)
        rows = [r for r in out.splitlines() if "|" in r]
        assert out.splitlines()[0].startswith("### f.txt (201 lines; changed regions")
        # Two hunks of context + change, not the whole file.
        assert len(rows) <= 2 * (2 * CHANGED_CONTEXT_LINES + 2)
        assert f"50|{_line_hash('line 50 changed')}|line 50 changed" in rows
        assert f"181|{_line_hash('line 180b')}|line 180b" in rows
        assert "...\n" in out  # hunk separator

    def test_identical_content_falls_back_to_full_render(self) -> None:
        from privibe.core.orchestrate.snapshot import render_changed_file

        text = "a\nb\n"
        assert render_changed_file("f", text, text) == render_file("f", text)
