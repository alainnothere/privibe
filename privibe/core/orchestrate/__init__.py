"""Manager/worker loop over a ledger on disk (the orchestrate tool).

One model plays every role, each role in a fresh context, and the roles
coordinate only through files the harness writes. The loop, the parsing, the
guards and the verifier are plain Python here; the model only ever answers one
question at a time.
"""

from __future__ import annotations

from privibe.core.orchestrate.ledger import (
    Checkpoint,
    Ledger,
    LedgerStateError,
    TaskItem,
)
from privibe.core.orchestrate.loop import (
    JobSpec,
    Outcome,
    OutcomeStatus,
    Progress,
    RoleCall,
    job_from_record,
    job_record,
    run_orchestration,
)
from privibe.core.orchestrate.snapshot import Bundle, BundleTooLargeError
from privibe.core.orchestrate.verifier import VerifierResult, run_verifier

__all__ = [
    "Bundle",
    "BundleTooLargeError",
    "Checkpoint",
    "JobSpec",
    "Ledger",
    "LedgerStateError",
    "Outcome",
    "OutcomeStatus",
    "Progress",
    "RoleCall",
    "TaskItem",
    "VerifierResult",
    "job_from_record",
    "job_record",
    "run_orchestration",
    "run_verifier",
]
