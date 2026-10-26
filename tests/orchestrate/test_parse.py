from __future__ import annotations

import pytest

from privibe.core.orchestrate.parse import (
    ParseError,
    extract_section,
    parse_manage,
    parse_plan,
    parse_worker_notes,
)


class TestParsePlan:
    def test_plain_shape(self) -> None:
        text = (
            "PLAN:\nFix the cache. It never invalidates.\n\n"
            "TASKS:\n1. Add an invalidation hook\n2. Add a test\n"
        )
        answer = parse_plan(text)
        assert answer.plan == "Fix the cache. It never invalidates."
        assert [t.text for t in answer.tasks] == [
            "Add an invalidation hook",
            "Add a test",
        ]
        assert all(not t.done for t in answer.tasks)

    def test_fenced_answer_is_unwrapped(self) -> None:
        text = "```\nPLAN:\nShort plan.\nTASKS:\n1. one\n```"
        answer = parse_plan(text)
        assert answer.plan == "Short plan."
        assert answer.tasks[0].text == "one"

    def test_missing_tasks_section(self) -> None:
        with pytest.raises(ParseError, match="TASKS"):
            parse_plan("PLAN:\nJust a plan and no list.")

    def test_tasks_without_numbered_lines(self) -> None:
        with pytest.raises(ParseError, match="numbered"):
            parse_plan("PLAN:\nplan\nTASKS:\n- bullet not number\n")

    def test_prose_before_tasks_counts_as_plan_without_label(self) -> None:
        answer = parse_plan("We should do X then Y.\nTASKS:\n1. X\n2) Y\n")
        assert answer.plan == "We should do X then Y."
        assert [t.id for t in answer.tasks] == [1, 2]


class TestParseManage:
    def test_next_id(self) -> None:
        text = "TASKS:\n1. [done] first\n2. [todo] second\n3. [todo] third\nNEXT: 2\n"
        answer = parse_manage(text)
        assert not answer.done
        assert answer.next_id == 2
        assert answer.next_task() is not None
        assert answer.next_task().text == "second"
        assert [t.done for t in answer.tasks] == [True, False, False]

    def test_done(self) -> None:
        answer = parse_manage("TASKS:\n1. [done] all\nNEXT: DONE")
        assert answer.done and answer.next_id is None

    def test_done_lowercase_with_period(self) -> None:
        assert parse_manage("1. [x] all\nnext: done.").done

    def test_next_accepts_task_prefix_and_hash(self) -> None:
        assert parse_manage("1. [todo] a\n2. [todo] b\nNEXT: task #2").next_id == 2

    def test_last_next_line_wins(self) -> None:
        text = "NEXT: 1\nTASKS:\n1. [todo] a\n2. [todo] b\nNEXT: 2\n"
        assert parse_manage(text).next_id == 2

    def test_missing_next(self) -> None:
        with pytest.raises(ParseError, match="NEXT"):
            parse_manage("TASKS:\n1. [todo] a\n")

    def test_next_names_unknown_task(self) -> None:
        with pytest.raises(ParseError, match="names no task"):
            parse_manage("1. [todo] a\nNEXT: 7")

    def test_next_names_done_task(self) -> None:
        with pytest.raises(ParseError, match="marked done"):
            parse_manage("1. [done] a\n2. [todo] b\nNEXT: 1")

    def test_next_garbage(self) -> None:
        with pytest.raises(ParseError, match="task id or DONE"):
            parse_manage("1. [todo] a\nNEXT: the second one")

    def test_duplicate_ids_keep_first(self) -> None:
        answer = parse_manage("1. [todo] a\n1. [todo] dup\n2. [todo] b\nNEXT: 2")
        assert [t.text for t in answer.tasks] == ["a", "b"]


class TestSections:
    def test_extract_section_stops_at_next_heading(self) -> None:
        text = "## Notes\nline one\nline two\n\n## Next steps\nnone\n"
        assert extract_section(text, "notes") == "line one\nline two"
        assert extract_section(text, "Next steps") == "none"
        assert extract_section(text, "Missing") is None

    def test_worker_notes_absent(self) -> None:
        assert parse_worker_notes("I did the thing.") is None

    def test_worker_notes_empty_section_is_none(self) -> None:
        assert parse_worker_notes("## Notes\n\n## Next steps\nnone") is None

    def test_worker_notes_present(self) -> None:
        assert parse_worker_notes("prose\n# Notes\nkept\n## Next steps\nx") == "kept"
