from __future__ import annotations

from backend.models.system import (
    AICase,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)


def build_baseline_cases() -> list[AICase]:
    cases = [
        # Temporary files
        (
            "temp-old",
            r"C:\Temp\old.tmp",
            FileCategory.TEMPORARY,
            50_000_000,
            90,
            False,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.LOW,
        ),
        (
            "temp-recent",
            r"C:\Temp\recent.tmp",
            FileCategory.TEMPORARY,
            5_000_000,
            2,
            False,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.MEDIUM,
        ),

        # Logs
        (
            "log-old",
            r"C:\Logs\application.log",
            FileCategory.LOG,
            100_000_000,
            120,
            False,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.LOW,
        ),

        # Crash dumps
        (
            "dump-old",
            r"C:\Dumps\crash.dmp",
            FileCategory.CRASH_DUMP,
            500_000_000,
            180,
            False,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.LOW,
        ),

        # User data
        (
            "user-document",
            r"C:\Users\Test\Documents\thesis.pdf",
            FileCategory.USER_DATA,
            50_000_000,
            365,
            False,
            True,
            False,
            False,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),

        (
            "user-photo",
            r"C:\Users\Test\Pictures\photo.jpg",
            FileCategory.USER_DATA,
            10_000_000,
            200,
            False,
            True,
            False,
            False,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),

        # Application data
        (
            "app-data",
            r"C:\Program Files\TestApp\data.bin",
            FileCategory.APPLICATION_DATA,
            200_000_000,
            300,
            False,
            False,
            True,
            False,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),

        # System data
        (
            "system-log",
            r"C:\Windows\Temp\system.etl",
            FileCategory.SYSTEM_DATA,
            20_000_000,
            90,
            True,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.HIGH,
        ),

        (
            "minidump",
            r"C:\Windows\Minidump\memory.dmp",
            FileCategory.CRASH_DUMP,
            100_000_000,
            60,
            True,
            False,
            False,
            False,
            RecommendationAction.REVIEW,
            RecommendationRisk.HIGH,
        ),

        # Locked file
        (
            "locked-temp",
            r"C:\Temp\locked.tmp",
            FileCategory.TEMPORARY,
            20_000_000,
            90,
            False,
            False,
            False,
            True,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),

        # Unknown files
        (
            "unknown-xyz",
            r"C:\somewhere\unknown.xyz",
            FileCategory.UNKNOWN,
            10_000_000,
            100,
            False,
            False,
            False,
            False,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),

        (
            "unknown-bin",
            r"C:\somewhere\data.bin",
            FileCategory.UNKNOWN,
            100_000_000,
            500,
            False,
            False,
            False,
            False,
            RecommendationAction.KEEP,
            RecommendationRisk.HIGH,
        ),
    ]

    return [
        AICase(
            case_id=case_id,
            context=DecisionContext(
                path=path,
                size=size,
                extension="." + path.rsplit(".", 1)[-1],
                category=category,
                exists=True,
                age_days=age_days,
                is_system_path=is_system_path,
                is_user_path=is_user_path,
                is_application_path=is_application_path,
                is_locked=is_locked,
                signals=[],
                current_action=action,
                current_risk=risk,
            ),
        )
        for (
            case_id,
            path,
            category,
            size,
            age_days,
            is_system_path,
            is_user_path,
            is_application_path,
            is_locked,
            action,
            risk,
        ) in cases
    ]