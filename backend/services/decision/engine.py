from __future__ import annotations

from backend.models.system import (
    DecisionContext,
    FileCategory,
    FileEvidence,
    FileRecord,
    Recommendation,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.context import context_to_file_and_evidence


POLICY_VERSION = "policy-v1"
"""
The version of the deterministic policy implemented in this module.

It exists because a label is meaningless without it. `data/train.jsonl` on
a training machine, a recorded agreement figure in a report, and a model
checkpoint are all statements about *some* policy, and until this constant
existed the only record of which one was prose in an ADR (§514/§516).

Bump it in the same commit that changes any verdict `make_recommendation`
emits, and publish `docs/policy-v<n>.md` alongside. Metrics measured under
different values are not comparable and must not be presented as a trend.
`docs/policy-v1.md` states the full procedure.
"""


def recommend_for_context(
    context: DecisionContext,
) -> Recommendation:
    """
    Re-derive the authoritative recommendation for a decision context.

    `DecisionContext` carries `current_action` / `current_risk` so that
    ground truth can travel with a case, but those fields are supplied by
    whoever built the case and are therefore not authority. Every caller
    that needs the deterministic verdict — the safety gate, the dataset
    labeler — must come through here so the verdict is always computed,
    never accepted (§28/§65/§443).
    """

    file, evidence = context_to_file_and_evidence(context)

    return make_recommendation(file, evidence)


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