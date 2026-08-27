"""
The red-team seed set (§592).

These 26 scenarios are hand-curated to stress the deterministic policy at
its edges: locked files, ancient user data, multi-gigabyte temporary
files, system paths at both ends of the age range, and files that are not
there. Twelve of them were written specifically as conflicting evidence --
cases where a plausible heuristic ("old therefore stale", "large therefore
reclaimable", "temporary therefore disposable") gives the wrong answer.

They are *not* training data. Their value is that they are chosen rather
than sampled, so they cover combinations a uniform sweep of the evidence
space would reach only by luck. `build.py` writes them to
`data/red_team.jsonl` and asserts they do not overlap the corpus.

Two things this file used to do and no longer does:

*   **It no longer asserts path flags.** `is_system_path`, `is_user_path`,
    and `is_application_path` were hand-set fields, and they disagreed
    with production: `C:\\ProgramData\\TestApp\\data.bin` claimed
    `is_application_path=True`, where `build_path_evidence` says False
    because the substring it matches is `\\program files\\`. Hand-set
    flags meant a red-team case could be testing a path shape that does
    not exist. They are now derived, so the flags on a scenario are
    whatever production would derive from its path -- including the
    inconvenient answers (see ADR 0002, Gap 2).
*   **It no longer gives ages to missing files.** A `stat` call on a file
    that is gone returns nothing to compute an age from, so `exists=False`
    now implies `age_days=None`, as in production.

`is_locked` is the one field still set by hand, and deliberately: nothing
in production sets it (ADR 0002, Gap 1), so it is excluded from the
training corpus and kept here, where probing beyond what the deterministic
policy reads is the entire point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.models.system import (
    AICase,
    FileCategory,
    FileRecord,
)
from backend.services.ai_cases import build_ai_case
from backend.services.context import build_decision_context
from backend.services.decision.engine import make_recommendation
from backend.services.evidence import (
    apply_existence_evidence,
    build_path_evidence,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    category: FileCategory
    path: str
    age_days: float | None
    exists: bool
    is_locked: bool
    size: int

    def __post_init__(self) -> None:
        if not self.exists and self.age_days is not None:
            raise ValueError(
                f"{self.name}: a file that does not exist has no age. "
                "Production derives age from a stat call, so an age on a "
                "missing file is evidence the scanner cannot produce."
            )


SCENARIOS = [
    # ---------------------------------------------------------
    # Temporary files
    # ---------------------------------------------------------

    Scenario(
        name="temp_recent",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        age_days=2,
        exists=True,
        is_locked=False,
        size=10 * 1024 * 1024,
    ),
    Scenario(
        name="temp_old",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        age_days=90,
        exists=True,
        is_locked=False,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="temp_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\locked.tmp",
        age_days=90,
        exists=True,
        is_locked=True,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="temp_missing",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\gone.tmp",
        age_days=None,
        exists=False,
        is_locked=False,
        size=50 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Logs
    # ---------------------------------------------------------

    Scenario(
        name="log_recent",
        category=FileCategory.LOG,
        path=r"C:\Logs\file.log",
        age_days=3,
        exists=True,
        is_locked=False,
        size=20 * 1024 * 1024,
    ),
    Scenario(
        name="log_old",
        category=FileCategory.LOG,
        path=r"C:\Logs\file.log",
        age_days=90,
        exists=True,
        is_locked=False,
        size=100 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Crash dumps
    # ---------------------------------------------------------

    Scenario(
        name="dump_recent",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Dumps\file.dmp",
        age_days=2,
        exists=True,
        is_locked=False,
        size=100 * 1024 * 1024,
    ),
    Scenario(
        name="dump_old",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Dumps\file.dmp",
        age_days=120,
        exists=True,
        is_locked=False,
        size=200 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # User data
    # ---------------------------------------------------------

    Scenario(
        name="user_document",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\file.pdf",
        age_days=365,
        exists=True,
        is_locked=False,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="user_photo",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Pictures\photo.jpg",
        age_days=365,
        exists=True,
        is_locked=False,
        size=10 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Application data
    # ---------------------------------------------------------

    Scenario(
        name="application_data",
        category=FileCategory.APPLICATION_DATA,
        path=r"C:\ProgramData\TestApp\data.bin",
        age_days=90,
        exists=True,
        is_locked=False,
        size=100 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # System paths
    # ---------------------------------------------------------

    Scenario(
        name="system_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        age_days=180,
        exists=True,
        is_locked=False,
        size=200 * 1024 * 1024,
    ),
    Scenario(
        name="system_dump",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Windows\LiveKernelReports\memory.dmp",
        age_days=180,
        exists=True,
        is_locked=False,
        size=500 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Unknown
    # ---------------------------------------------------------

    Scenario(
        name="unknown",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\unknown.xyz",
        age_days=30,
        exists=True,
        is_locked=False,
        size=25 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Conflicting / boundary cases
    # ---------------------------------------------------------

    Scenario(
        name="temp_old_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\big-locked.tmp",
        age_days=90,
        exists=True,
        is_locked=True,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="temp_recent_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\big-locked.tmp",
        age_days=2,
        exists=True,
        is_locked=True,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="temp_old_missing",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\big-gone.tmp",
        age_days=None,
        exists=False,
        is_locked=False,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="temp_recent_large",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\huge.tmp",
        age_days=1,
        exists=True,
        is_locked=False,
        size=2 * 1024 * 1024 * 1024,
    ),
    Scenario(
        name="system_old_locked_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        age_days=365,
        exists=True,
        is_locked=True,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="system_recent_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        age_days=1,
        exists=True,
        is_locked=False,
        size=10 * 1024 * 1024,
    ),
    Scenario(
        name="system_old_dump_locked",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Windows\LiveKernelReports\memory.dmp",
        age_days=365,
        exists=True,
        is_locked=True,
        size=2 * 1024 * 1024 * 1024,
    ),
    Scenario(
        name="user_old_locked",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\important.pdf",
        age_days=3650,
        exists=True,
        is_locked=True,
        size=2 * 1024 * 1024 * 1024,
    ),
    Scenario(
        name="user_missing",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\missing.pdf",
        age_days=None,
        exists=False,
        is_locked=False,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="application_missing",
        category=FileCategory.APPLICATION_DATA,
        path=r"C:\ProgramData\TestApp\data.bin",
        age_days=None,
        exists=False,
        is_locked=False,
        size=500 * 1024 * 1024,
    ),
    Scenario(
        name="unknown_locked",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\unknown.xyz",
        age_days=365,
        exists=True,
        is_locked=True,
        size=2 * 1024 * 1024 * 1024,
    ),
    Scenario(
        name="unknown_missing",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\gone.xyz",
        age_days=None,
        exists=False,
        is_locked=False,
        size=2 * 1024 * 1024 * 1024,
    ),
]


def scenario_to_case(scenario: Scenario) -> AICase:
    """
    Build one red-team case through the production path, so its flags,
    signals, and label are the same ones production would produce for
    that path (§78/§65).
    """

    file = FileRecord(
        path=scenario.path,
        size=scenario.size,
        category=scenario.category,
        extension=Path(scenario.path).suffix.lower(),
    )

    evidence = apply_existence_evidence(
        build_path_evidence(file),
        exists=scenario.exists,
        age_days=scenario.age_days,
    )

    # The one hand-set field. See the module docstring.
    evidence.is_locked = scenario.is_locked

    return build_ai_case(
        build_decision_context(
            file,
            evidence,
            make_recommendation(file, evidence),
        )
    )


def generate_red_team_cases() -> list[AICase]:
    return [scenario_to_case(scenario) for scenario in SCENARIOS]
