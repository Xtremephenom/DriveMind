from __future__ import annotations

from backend.models.system import (
    AICase,
    FileEvidence,
    FileRecord,
)

from backend.services.decision.engine import make_recommendation


def label_case(case: AICase):
    """
    Produce the trusted DriveMind label for a synthetic case.

    The deterministic decision engine is the single source
    of truth for the expected action and risk.
    """

    file = FileRecord(
        path=case.context.path,
        size=case.context.size,
        category=case.context.category,
        extension=case.context.extension,
    )

    evidence = FileEvidence(
        path=case.context.path,
        size=case.context.size,
        extension=case.context.extension,
        age_days=case.context.age_days,
        exists=case.context.exists,
        is_locked=case.context.is_locked,
        is_system_path=case.context.is_system_path,
        is_user_path=case.context.is_user_path,
        is_application_path=case.context.is_application_path,
        category=case.context.category,
        signals=case.context.signals,
    )

    recommendation = make_recommendation(
        file,
        evidence,
    )

    return recommendation