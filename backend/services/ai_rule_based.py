from __future__ import annotations

from backend.models.system import (
    AICase,
    AIResponse,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai import AIProvider


class RuleBasedAIProvider(AIProvider):
    """
    Temporary deterministic AI provider.

    This is NOT a machine-learning model.

    It exists only so the AI interface can be tested
    before a real local model is introduced.
    """

    def analyze(self, case: AICase) -> AIResponse:
        context = case.context

        if context.current_action == RecommendationAction.KEEP:
            return AIResponse(
                case_id=case.case_id,
                action=RecommendationAction.KEEP,
                risk=RecommendationRisk.HIGH,
                confidence=0.99,
                explanation=(
                    "The current deterministic analysis does not "
                    "provide enough evidence for cleanup."
                ),
            )

        if context.current_risk == RecommendationRisk.LOW:
            return AIResponse(
                case_id=case.case_id,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                confidence=0.90,
                explanation=(
                    "The available evidence indicates that this "
                    "file may be suitable for cleanup review."
                ),
            )

        return AIResponse(
            case_id=case.case_id,
            action=RecommendationAction.REVIEW,
            risk=context.current_risk,
            confidence=0.75,
            explanation=(
                "The file requires human review before any "
                "cleanup decision is made."
            ),
        )