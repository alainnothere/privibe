from __future__ import annotations

from pathlib import Path
import sys

import pytest

from privibe.core.orchestrate.verifier import run_verifier, tail_chars


@pytest.mark.asyncio
async def test_pass_and_output(tmp_path: Path) -> None:
    result = await run_verifier(
        f'"{sys.executable}" -c "print(\'ok\')"',
        cwd=tmp_path,
        timeout_s=30,
        output_chars=100,
    )
    assert result.passed and result.exit_code == 0
    assert result.output.strip() == "ok"


@pytest.mark.asyncio
async def test_fail_keeps_stderr_tail(tmp_path: Path) -> None:
    cmd = f"\"{sys.executable}\" -c \"import sys; print('x'*500); print('last', file=sys.stderr); sys.exit(3)\""
    result = await run_verifier(cmd, cwd=tmp_path, timeout_s=30, output_chars=60)
    assert not result.passed and result.exit_code == 3
    assert result.output.startswith("[... first part cut")
    assert "last" in result.output


@pytest.mark.asyncio
async def test_timeout(tmp_path: Path) -> None:
    cmd = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    result = await run_verifier(cmd, cwd=tmp_path, timeout_s=0.5, output_chars=60)
    assert result.timed_out and not result.passed
    assert "timed out" in result.output


def test_tail_chars() -> None:
    assert tail_chars("abc", 5) == "abc"
    assert tail_chars("abcdef", 3).endswith("def")
