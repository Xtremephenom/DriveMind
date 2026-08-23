from __future__ import annotations

from backend.models.system import (
    DecisionContext,
    FileEvidence,
    FileRecord,
    Recommendation,
)


def build_decision_context(
    file: FileRecord,
    evidence: FileEvidence,
    recommendation: Recommendation,
) -> DecisionContext:
    """
    Combine file metadata, evidence, and the current
    deterministic recommendation into one decision context.

    This function is read-only.
    """

    return DecisionContext(
        path=file.path,
        size=file.size,
        extension=evidence.extension,
        category=file.category,

        exists=evidence.exists,
        age_days=evidence.age_days,

        is_system_path=evidence.is_system_path,
        is_user_path=evidence.is_user_path,
        is_application_path=evidence.is_application_path,
        is_locked=evidence.is_locked,

        signals=list(evidence.signals),

        current_action=recommendation.action,
        current_risk=recommendation.risk,
    )


def context_to_dict(
    context: DecisionContext,
) -> dict:
    return {
        "path": context.path,
        "size": context.size,
        "extension": context.extension,
        "category": context.category.value,

        "exists": context.exists,
        "age_days": context.age_days,

        "is_system_path": context.is_system_path,
        "is_user_path": context.is_user_path,
        "is_application_path": context.is_application_path,
        "is_locked": context.is_locked,

        "signals": context.signals,

        "current_action": context.current_action.value,
        "current_risk": context.current_risk.value,
    }