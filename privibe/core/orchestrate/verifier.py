from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys


@dataclass(frozen=True)
class VerifierResult:
    exit_code: int
    output: str
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def tail_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "[... first part cut, last part kept ...]\n" + text[-limit:]


async def run_verifier(
    command: str, *, cwd: Path, timeout_s: float, output_chars: int
) -> VerifierResult:
    """Run the job's done-command through the shell, in `cwd`, and report the
    exit code with the tail of its combined output. Ground truth for the loop:
    a failing run vetoes a manager's DONE.
    """
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Own process group, so a timeout kills the shell's children too: they
        # hold the pipe open and asyncio waits for it to close.
        start_new_session=sys.platform != "win32",
    )
    try:
        raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        if sys.platform == "win32":
            proc.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        await proc.wait()
        return VerifierResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            output=f"verifier timed out after {timeout_s:g}s",
            timed_out=True,
        )
    text = raw.decode("utf-8", errors="replace")
    return VerifierResult(
        exit_code=proc.returncode if proc.returncode is not None else -1,
        output=tail_chars(text, output_chars),
    )
