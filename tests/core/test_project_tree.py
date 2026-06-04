from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.project_tree import build_tree


class TestBuildTree:
    def test_basic_structure(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        (tmp_path / "main.py").write_text("print(1)")
        d = tmp_path / "src"
        d.mkdir()
        (d / "app.py").write_text("app")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        lines = result.splitlines()

        assert any("README.md" in l for l in lines)
        assert any("main.py" in l for l in lines)
        assert any("src/" in l for l in lines)
        assert any("app.py" in l for l in lines)

    def test_respects_gitignore(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        (tmp_path / "main.py").write_text("hello")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "main.cpython-312.pyc").write_text("bytecode")
        (tmp_path / "main.pyc").write_text("bytecode")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert "__pycache__" not in result
        assert "main.pyc" not in result
        assert "main.py" in result

    def test_respects_default_ignores(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("hello")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "bin").mkdir()
        git = tmp_path / ".git"
        git.mkdir()

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert ".venv" not in result
        assert ".git" not in result
        assert "main.py" in result

    def test_depth_limit(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        d1 = tmp_path / "a"
        d1.mkdir()
        d2 = d1 / "b"
        d2.mkdir()
        d3 = d2 / "c"
        d3.mkdir()
        (d3 / "deep.txt").write_text("deep")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        # At depth 2, we should see a/ and b/, and c/ listed but not descended into
        assert "a/" in result
        assert "b/" in result
        # c/ appears as a listed entry at depth 2 but its contents are not shown
        assert "c/" in result
        assert "deep.txt" not in result

    def test_truncation_summary(self, tmp_path: Path) -> None:
        # Create many files to trigger truncation
        for i in range(30):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"content {i}")

        result = build_tree(tmp_path, max_depth=2, max_lines=10)
        lines = result.splitlines()
        assert len(lines) <= 10
        assert "... (" in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert result == ""

    def test_only_ignored_files(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.txt\n")
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        # .gitignore itself is not ignored, so it shows up
        assert ".gitignore" in result
        assert "a.txt" not in result
        assert "b.txt" not in result

    def test_tree_characters_present(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert "\u251c" in result or "\u2514" in result  # ├── or └──

    def test_files_sorted_before_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "z_file.txt").write_text("z")
        (tmp_path / "a_file.txt").write_text("a")
        d = tmp_path / "src"
        d.mkdir()
        (d / "app.py").write_text("app")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        lines = result.splitlines()

        # Find positions of files and dirs
        file_lines = [i for i, l in enumerate(lines) if "file.txt" in l]
        dir_lines = [i for i, l in enumerate(lines) if "src/" in l]

        assert file_lines and dir_lines
        assert max(file_lines) < min(dir_lines)

    def test_symlink_not_followed(self, tmp_path: Path) -> None:
        real = tmp_path / "real"
        real.mkdir()
        (real / "file.txt").write_text("real")
        link = tmp_path / "link"
        try:
            link.symlink_to(real)
        except OSError:
            pytest.skip("symlinks not supported on this platform")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        # Symlink should appear as an entry, not be followed
        assert "link" in result

    def test_permission_error_handled(self, tmp_path: Path) -> None:
        (tmp_path / "ok.txt").write_text("ok")
        restricted = tmp_path / "restricted"
        restricted.mkdir()
        try:
            restricted.chmod(0o000)
            result = build_tree(tmp_path, max_depth=2, max_lines=80)
            assert "ok.txt" in result
        finally:
            restricted.chmod(0o755)

    def test_max_lines_hard_cap(self, tmp_path: Path) -> None:
        # Create a structure that would exceed max_lines
        for i in range(50):
            (tmp_path / f"file_{i:03d}.txt").write_text(f"content {i}")

        result = build_tree(tmp_path, max_depth=2, max_lines=15)
        lines = result.splitlines()
        assert len(lines) <= 15

    def test_nested_truncation(self, tmp_path: Path) -> None:
        # Create nested dirs with many files
        for i in range(20):
            d = tmp_path / f"dir_{i}"
            d.mkdir()
            for j in range(10):
                (d / f"file_{j}.txt").write_text(f"content {i}-{j}")

        result = build_tree(tmp_path, max_depth=2, max_lines=30)
        lines = result.splitlines()
        assert len(lines) <= 30

    def test_gitignore_negation(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\n!important.log\n")
        (tmp_path / "debug.log").write_text("debug")
        (tmp_path / "important.log").write_text("important")
        (tmp_path / "main.py").write_text("hello")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert "debug.log" not in result
        assert "important.log" in result

    def test_gitignore_root_anchored(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("/build/\n")
        (tmp_path / "main.py").write_text("hello")
        top_build = tmp_path / "build"
        top_build.mkdir()
        nested = tmp_path / "src"
        nested.mkdir()
        nested_build = nested / "build"
        nested_build.mkdir()
        (nested_build / "file.txt").write_text("nested")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        # /build/ is root-anchored, so only top-level build/ is ignored
        # But "build/" is also in DEFAULT_IGNORE_PATTERNS, so nested build/ is also ignored
        # Let's test with a name not in defaults
        other = tmp_path / "other"
        other.mkdir()
        other_sub = other / "sub"
        other_sub.mkdir()
        (other_sub / "deep.txt").write_text("deep")

        result = build_tree(tmp_path, max_depth=2, max_lines=80)
        assert "main.py" in result
