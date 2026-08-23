from __future__ import annotations

from backend.models.system import (
    FileCategory,
    FileEvidence,
    FileRecord,
    Recommendation,
    RecommendationAction,
    RecommendationRisk,
)


def make_recommendation(
    file: FileRecord,
    evidence: FileEvidence,
) -> Recommendation:
    """
    Make a conservative cleanup recommendation.

    The engine is deterministic and read-only.
    It does not modify the filesystem.
    """

    category = file.category

    # ---------------------------------------------------------
    # Unknown files
    # ---------------------------------------------------------

    if category == FileCategory.UNKNOWN:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.KEEP,
            risk=RecommendationRisk.HIGH,
            reason=(
                "The file could not be confidently classified. "
                "DriveMind will not recommend automatic removal."
            ),
        )

    # ---------------------------------------------------------
    # User data
    # ---------------------------------------------------------

    if category == FileCategory.USER_DATA:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.KEEP,
            risk=RecommendationRisk.HIGH,
            reason=(
                "The file appears to contain user data. "
                "DriveMind will not recommend automatic removal."
            ),
        )

    # ---------------------------------------------------------
    # Application data
    # ---------------------------------------------------------

    if category == FileCategory.APPLICATION_DATA:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.KEEP,
            risk=RecommendationRisk.HIGH,
            reason=(
                "The file may belong to an installed application. "
                "DriveMind will not recommend automatic removal."
            ),
        )

    # ---------------------------------------------------------
    # System path safety gate
    # ---------------------------------------------------------

    if evidence.is_system_path:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.REVIEW,
            risk=RecommendationRisk.HIGH,
            reason=(
                "The file is located inside the Windows directory. "
                "Additional verification is required before cleanup."
            ),
        )

    # ---------------------------------------------------------
    # Missing files
    # ---------------------------------------------------------

    if not evidence.exists:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.KEEP,
            risk=RecommendationRisk.HIGH,
            reason=(
                "The file no longer exists. "
                "No cleanup action is required."
            ),
        )

    # ---------------------------------------------------------
    # Temporary files
    # ---------------------------------------------------------

    if category == FileCategory.TEMPORARY:

        if evidence.age_days is not None and evidence.age_days >= 30:
            return Recommendation(
                path=file.path,
                size=file.size,
                category=category,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                reason=(
                    "The file is classified as temporary data "
                    "and has not been modified for at least 30 days. "
                    "It is a strong candidate for cleanup review."
                ),
            )

        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.REVIEW,
            risk=RecommendationRisk.MEDIUM,
            reason=(
                "The file appears to be temporary data, "
                "but it may still be relatively recent."
            ),
        )

    # ---------------------------------------------------------
    # Logs
    # ---------------------------------------------------------

    if category == FileCategory.LOG:

        if evidence.age_days is not None and evidence.age_days >= 30:
            return Recommendation(
                path=file.path,
                size=file.size,
                category=category,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                reason=(
                    "The file appears to be an old log. "
                    "It may be removable after confirming that "
                    "it is no longer needed for troubleshooting."
                ),
            )

        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.REVIEW,
            risk=RecommendationRisk.MEDIUM,
            reason=(
                "The file appears to be a log and may still "
                "be useful for troubleshooting."
            ),
        )

    # ---------------------------------------------------------
    # Crash dumps
    # ---------------------------------------------------------

    if category == FileCategory.CRASH_DUMP:

        if evidence.age_days is not None and evidence.age_days >= 30:
            return Recommendation(
                path=file.path,
                size=file.size,
                category=category,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                reason=(
                    "The crash dump is older than 30 days. "
                    "It may be a strong cleanup candidate if "
                    "diagnostic history is no longer required."
                ),
            )

        return Recommendation(
            path=file.path,
            size=file.size,
            category=category,
            action=RecommendationAction.REVIEW,
            risk=RecommendationRisk.MEDIUM,
            reason=(
                "The crash dump may still be useful for "
                "diagnosing a recent system problem."
            ),
        )

    # ---------------------------------------------------------
    # Conservative fallback
    # ---------------------------------------------------------

    return Recommendation(
        path=file.path,
        size=file.size,
        category=category,
        action=RecommendationAction.REVIEW,
        risk=RecommendationRisk.MEDIUM,
        reason=(
            "The file is classified but does not have a "
            "sufficiently strong cleanup rule."
        ),
    )