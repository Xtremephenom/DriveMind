from __future__ import annotations

from backend.models.system import AICase, Recommendation
from backend.services.decision.engine import recommend_for_context


def label_case(case: AICase) -> Recommendation:
    """
    Produce the trusted DriveMind label for a synthetic case.

    The deterministic decision engine is the single source
    of truth for the expected action and risk (§65).
    """

    return recommend_for_context(case.context)
