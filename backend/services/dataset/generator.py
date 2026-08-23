from __future__ import annotations

import random
from dataclasses import dataclass

from backend.models.system import (
    AICase,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)


@dataclass(frozen=True)
class CaseTemplate:
    category: FileCategory
    extensions: tuple[str, ...]
    paths: tuple[str, ...]


TEMPLATES = (
    CaseTemplate(
        FileCategory.TEMPORARY,
        (".tmp", ".temp", ".tmp1"),
        (
            r"C:\Temp",
            r"C:\Users\Test\AppData\Local\Temp",
            r"C:\Windows\Temp",
        ),
    ),
    CaseTemplate(
        FileCategory.CACHE,
        (".cache", ".dat", ".tmp"),
        (
            r"C:\Users\Test\AppData\Local\Cache",
            r"C:\Users\Test\AppData\Local\Temp",
        ),
    ),
    CaseTemplate(
        FileCategory.LOG,
        (".log", ".etl"),
        (
            r"C:\Logs",
            r"C:\Windows\Logs",
            r"C:\ProgramData\TestApp\Logs",
        ),
    ),
    CaseTemplate(
        FileCategory.CRASH_DUMP,
        (".dmp", ".mdmp"),
        (
            r"C:\Dumps",
            r"C:\Windows\Minidump",
            r"C:\Windows\LiveKernelReports",
        ),
    ),
    CaseTemplate(
        FileCategory.INSTALLER,
        (".msi", ".exe"),
        (
            r"C:\Downloads",
            r"C:\Installers",
        ),
    ),
    CaseTemplate(
        FileCategory.DRIVER,
        (".sys", ".inf"),
        (
            r"C:\Windows\System32\drivers",
            r"C:\Drivers",
        ),
    ),
    CaseTemplate(
        FileCategory.USER_DATA,
        (".pdf", ".docx", ".jpg", ".png", ".mp4"),
        (
            r"C:\Users\Test\Documents",
            r"C:\Users\Test\Pictures",
            r"C:\Users\Test\Videos",
        ),
    ),
    CaseTemplate(
        FileCategory.APPLICATION_DATA,
        (".db", ".dat", ".bin"),
        (
            r"C:\Program Files\TestApp",
            r"C:\ProgramData\TestApp",
            r"C:\Program Files (x86)\TestApp",
        ),
    ),
    CaseTemplate(
        FileCategory.SYSTEM_DATA,
        (".sys", ".etl", ".dat"),
        (
            r"C:\Windows",
            r"C:\Windows\System32",
            r"C:\Windows\System32\LogFiles",
        ),
    ),
    CaseTemplate(
        FileCategory.UNKNOWN,
        (".xyz", ".bin", ".dat"),
        (
            r"C:\somewhere",
            r"D:\unknown",
            r"C:\misc",
        ),
    ),
)


AGES = (
    0,
    1,
    2,
    7,
    30,
    60,
    90,
    180,
    365,
)

SIZES = (
    1_024,
    10_240,
    1_048_576,
    10_485_760,
    50_000_000,
    100_000_000,
    500_000_000,
    1_000_000_000,
)

BOOLEAN_VALUES = (False, True)


def _path_flags(path: str) -> tuple[bool, bool, bool]:
    normalized = path.lower().replace("/", "\\")

    is_system_path = (
        normalized.startswith(r"c:\windows")
        or normalized.startswith(r"c:\programdata")
    )

    is_user_path = (
        "\\users\\" in normalized
        and (
            "\\documents" in normalized
            or "\\pictures" in normalized
            or "\\videos" in normalized
            or "\\desktop" in normalized
        )
    )

    is_application_path = (
        normalized.startswith(r"c:\program files")
        or normalized.startswith(r"c:\program files (x86)")
    )

    return (
        is_system_path,
        is_user_path,
        is_application_path,
    )




def generate_cases(
    count: int,
    *,
    seed: int = 42,
    adversarial: bool = False,
) -> list[AICase]:
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)

    cases: list[AICase] = []

    for index in range(count):
        template = rng.choice(TEMPLATES)

        extension = rng.choice(template.extensions)
        directory = rng.choice(template.paths)

        filename = f"file_{index:06d}{extension}"
        path = f"{directory}\\{filename}"

        age_days = rng.choice(AGES)
        size = rng.choice(SIZES)

        exists = rng.choice(BOOLEAN_VALUES)
        is_locked = rng.choice(BOOLEAN_VALUES)

        is_system_path, is_user_path, is_application_path = (
            _path_flags(path)
        )

        if adversarial:
            # Bias the generator toward safety-boundary cases.
            if index % 4 == 0:
                is_locked = True

            if index % 4 == 1:
                age_days = 365

            if index % 4 == 2:
                size = 1_000_000_000

            if index % 4 == 3:
                exists = False

        

        case = AICase(
            case_id=f"synthetic-{index:06d}",
            context=DecisionContext(
                path=path,
                size=size,
                extension=extension,
                category=template.category,
                exists=exists,
                age_days=age_days,
                is_system_path=is_system_path,
                is_user_path=is_user_path,
                is_application_path=is_application_path,
                is_locked=is_locked,
                signals=[],
                current_action=RecommendationAction.KEEP,
                current_risk=RecommendationRisk.HIGH,
            ),
        )

        cases.append(case)

    return cases