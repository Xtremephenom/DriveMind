"""
Deterministic stand-in for a real AI provider.

This is NOT a machine-learning model, and it is NOT a baseline. It mirrors
the deterministic engine, so its agreement with the engine is 1.0 by
construction and its unsafe-escalation rate is 0 by construction. Quoting
either number as a model result would be a false claim (§555). It exists
so the AI interface, the safety gate, and the fallback path can be
exercised before a local model is introduced.

Like every other consumer of a case, it derives the deterministic verdict
rather than reading `current_action` / `current_risk` off the context. A
stand-in that trusts the ground truth travelling with a case would agree
with a *tampered* case too, which would make it useless for testing the
gate.
"""

from __future__ import annotations

from backend.models.system import (
    AICase,
    AIResponse,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai import AIProvider
from backend.services.decision.engine import recommend_for_context


class RuleBasedAIProvider(AIProvider):

    def analyze(self, case: AICase) -> AIResponse:
        deterministic = recommend_for_context(case.context)

        if deterministic.action == RecommendationAction.KEEP:
            return AIResponse(
                case_id=case.case_id,
                action=RecommendationAction.KEEP,
                risk=RecommendationRisk.HIGH,
                explanation=(
                    "The current deterministic analysis does not "
                    "provide enough evidence for cleanup."
                ),
            )

        if deterministic.risk == RecommendationRisk.LOW:
            return AIResponse(
                case_id=case.case_id,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                explanation=(
                    "The available evidence indicates that this "
                    "file may be suitable for cleanup review."
                ),
            )

        return AIResponse(
            case_id=case.case_id,
            action=RecommendationAction.REVIEW,
            risk=deterministic.risk,
            explanation=(
                "The file requires human review before any "
                "cleanup decision is made."
            ),
        )
