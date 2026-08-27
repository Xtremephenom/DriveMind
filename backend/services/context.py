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


def context_to_file_and_evidence(
    context: DecisionContext,
) -> tuple[FileRecord, FileEvidence]:
    """
    Reverse of `build_decision_context`.

    Recovers the evidence a context was built from, deliberately
    dropping `current_action` / `current_risk`: those are a *conclusion*,
    and anything that needs the conclusion must re-derive it from this
    evidence rather than trust the caller's copy (§443).
    """

    file = FileRecord(
        path=context.path,
        size=context.size,
        category=context.category,
        extension=context.extension,
    )

    evidence = FileEvidence(
        path=context.path,
        size=context.size,
        extension=context.extension,
        age_days=context.age_days,
        exists=context.exists,
        is_locked=context.is_locked,
        is_system_path=context.is_system_path,
        is_user_path=context.is_user_path,
        is_application_path=context.is_application_path,
        category=context.category,
        signals=list(context.signals),
    )

    return file, evidence


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