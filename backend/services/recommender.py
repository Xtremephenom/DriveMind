from __future__ import annotations

from backend.models.system import (
    FileCategory,
    FileRecord,
    Recommendation,
    RecommendationAction,
    RiskLevel,
)


def recommend(file: FileRecord) -> Recommendation:
    """
    Generate a conservative recommendation for a file.

    This function is read-only.
    It never deletes or modifies anything.
    """

    if file.category == FileCategory.TEMPORARY:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.REVIEW,
            risk=RiskLevel.MEDIUM,
            reason=(
                "Temporary file. It may be removable, "
                "but its current usage must be verified first."
            ),
        )

    if file.category == FileCategory.CACHE:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.REVIEW,
            risk=RiskLevel.MEDIUM,
            reason=(
                "Application cache. It may be regenerated, "
                "but application-specific behavior should be checked."
            ),
        )

    if file.category == FileCategory.CRASH_DUMP:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.REVIEW,
            risk=RiskLevel.LOW,
            reason=(
                "Crash diagnostic data. It may be removable if "
                "historical diagnostic information is no longer needed."
            ),
        )

    if file.category == FileCategory.LOG:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.REVIEW,
            risk=RiskLevel.MEDIUM,
            reason=(
                "Log file. It may contain information useful for "
                "troubleshooting and should be reviewed before removal."
            ),
        )

    if file.category == FileCategory.INSTALLER:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.KEEP,
            risk=RiskLevel.HIGH,
            reason=(
                "Installer cache may be required for application "
                "repair, update, or uninstall operations."
            ),
        )

    if file.category == FileCategory.DRIVER:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.KEEP,
            risk=RiskLevel.HIGH,
            reason=(
                "Driver package may be required for hardware recovery "
                "or driver reinstallation."
            ),
        )

    if file.category == FileCategory.SYSTEM_DATA:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.KEEP,
            risk=RiskLevel.HIGH,
            reason=(
                "System software data. Removing it could affect "
                "Windows or installed software."
            ),
        )

    if file.category == FileCategory.APPLICATION_DATA:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.REVIEW,
            risk=RiskLevel.HIGH,
            reason=(
                "Application data. Removing it may reset or damage "
                "application state."
            ),
        )

    if file.category == FileCategory.USER_DATA:
        return Recommendation(
            path=file.path,
            size=file.size,
            category=file.category,
            action=RecommendationAction.KEEP,
            risk=RiskLevel.HIGH,
            reason=(
                "User data should not be removed automatically."
            ),
        )

    return Recommendation(
        path=file.path,
        size=file.size,
        category=file.category,
        action=RecommendationAction.KEEP,
        risk=RiskLevel.HIGH,
        reason=(
            "The file could not be confidently classified. "
            "DriveMind will not recommend automatic removal."
        ),
    )


def recommendation_to_dict(
    recommendation: Recommendation,
) -> dict:
    return {
        "path": recommendation.path,
        "size": recommendation.size,
        "category": recommendation.category.value,
        "action": recommendation.action.value,
        "risk": recommendation.risk.value,
        "reason": recommendation.reason,
    }