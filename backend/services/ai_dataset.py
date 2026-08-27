"""
The 12-case AI baseline fixture.

Every case is built through the production path — `make_recommendation`
then `build_decision_context` — so the ground truth carried in
`current_action` / `current_risk` is always the engine's own verdict and
never a hand-written one (§65).

This used to be a table of hand-written label pairs. It disagreed with
the engine: `locked-temp` was recorded as KEEP/HIGH while the engine
returns REVIEW/LOW, so every agreement figure ever measured against this
fixture was scored partly against a wrong answer. Building the labels
instead of asserting them makes that class of drift impossible.
"""

from __future__ import annotations

from backend.models.system import (
    AICase,
    FileCategory,
    FileEvidence,
    FileRecord,
)
from backend.services.context import build_decision_context
from backend.services.decision.engine import make_recommendation


_BASELINE_SPECS: tuple[dict, ...] = (
    # Temporary files
    {
        "case_id": "temp-old",
        "path": r"C:\Temp\old.tmp",
        "category": FileCategory.TEMPORARY,
        "size": 50_000_000,
        "age_days": 90,
    },
    {
        "case_id": "temp-recent",
        "path": r"C:\Temp\recent.tmp",
        "category": FileCategory.TEMPORARY,
        "size": 5_000_000,
        "age_days": 2,
    },

    # Logs
    {
        "case_id": "log-old",
        "path": r"C:\Logs\application.log",
        "category": FileCategory.LOG,
        "size": 100_000_000,
        "age_days": 120,
    },

    # Crash dumps
    {
        "case_id": "dump-old",
        "path": r"C:\Dumps\crash.dmp",
        "category": FileCategory.CRASH_DUMP,
        "size": 500_000_000,
        "age_days": 180,
    },

    # User data
    {
        "case_id": "user-document",
        "path": r"C:\Users\Test\Documents\thesis.pdf",
        "category": FileCategory.USER_DATA,
        "size": 50_000_000,
        "age_days": 365,
        "is_user_path": True,
    },
    {
        "case_id": "user-photo",
        "path": r"C:\Users\Test\Pictures\photo.jpg",
        "category": FileCategory.USER_DATA,
        "size": 10_000_000,
        "age_days": 200,
        "is_user_path": True,
    },

    # Application data
    {
        "case_id": "app-data",
        "path": r"C:\Program Files\TestApp\data.bin",
        "category": FileCategory.APPLICATION_DATA,
        "size": 200_000_000,
        "age_days": 300,
        "is_application_path": True,
    },

    # System data
    {
        "case_id": "system-log",
        "path": r"C:\Windows\Temp\system.etl",
        "category": FileCategory.SYSTEM_DATA,
        "size": 20_000_000,
        "age_days": 90,
        "is_system_path": True,
    },
    {
        "case_id": "minidump",
        "path": r"C:\Windows\Minidump\memory.dmp",
        "category": FileCategory.CRASH_DUMP,
        "size": 100_000_000,
        "age_days": 60,
        "is_system_path": True,
    },

    # Locked file
    #
    # `is_locked` is collected nowhere and read nowhere under policy-v1,
    # so this case is currently indistinguishable from `temp-old` to the
    # engine. That is the gap, recorded in ADR 0002 — not something this
    # fixture may paper over with a label the engine does not produce.
    {
        "case_id": "locked-temp",
        "path": r"C:\Temp\locked.tmp",
        "category": FileCategory.TEMPORARY,
        "size": 20_000_000,
        "age_days": 90,
        "is_locked": True,
    },

    # Unknown files
    {
        "case_id": "unknown-xyz",
        "path": r"C:\somewhere\unknown.xyz",
        "category": FileCategory.UNKNOWN,
        "size": 10_000_000,
        "age_days": 100,
    },
    {
        "case_id": "unknown-bin",
        "path": r"C:\somewhere\data.bin",
        "category": FileCategory.UNKNOWN,
        "size": 100_000_000,
        "age_days": 500,
    },
)


def build_baseline_cases() -> list[AICase]:
    """
    Build the baseline evaluation cases, labels included.

    The `case_id` values are stable, human-readable names so a baseline
    report can be read and compared by hand. Generated dataset rows use
    the content-addressed `build_case_id` instead.
    """

    return [_build_case(**spec) for spec in _BASELINE_SPECS]


def _build_case(
    *,
    case_id: str,
    path: str,
    category: FileCategory,
    size: int,
    age_days: float,
    exists: bool = True,
    is_system_path: bool = False,
    is_user_path: bool = False,
    is_application_path: bool = False,
    is_locked: bool = False,
) -> AICase:

    extension = "." + path.rsplit(".", 1)[-1]

    file = FileRecord(
        path=path,
        size=size,
        category=category,
        extension=extension,
    )

    evidence = FileEvidence(
        path=path,
        size=size,
        extension=extension,
        age_days=age_days,
        exists=exists,
        is_locked=is_locked,
        is_system_path=is_system_path,
        is_user_path=is_user_path,
        is_application_path=is_application_path,
        category=category,
        signals=[],
    )

    return AICase(
        case_id=case_id,
        context=build_decision_context(
            file,
            evidence,
            make_recommendation(file, evidence),
        ),
    )
