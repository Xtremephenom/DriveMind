from __future__ import annotations

from backend.models.system import (
    AICase,
    AIResponse,
    RecommendationAction,
    RecommendationRisk,
)


_ACTION_RANK = {
    RecommendationAction.KEEP: 0,
    RecommendationAction.REVIEW: 1,
    RecommendationAction.DELETE: 2,
}


def validate_ai_response(
    case: AICase,
    response: AIResponse,
) -> AIResponse:
    """
    Validate an AI response against DriveMind's
    deterministic safety decision.

    The AI may not escalate the permitted action
    or reduce the deterministic risk level.
    """

    allowed_action = case.context.current_action
    allowed_risk = case.context.current_risk

    # Never allow the AI to escalate the action.
    if _ACTION_RANK[response.action] > _ACTION_RANK[allowed_action]:
        return AIResponse(
            case_id=response.case_id,
            action=allowed_action,
            risk=allowed_risk,
            confidence=response.confidence,
            explanation=(
                "AI recommendation was more permissive than "
                "DriveMind's deterministic safety policy. "
                "The deterministic recommendation was enforced."
            ),
        )

    # Never allow the AI to downgrade the deterministic risk.
    risk_rank = {
        RecommendationRisk.LOW: 0,
        RecommendationRisk.MEDIUM: 1,
        RecommendationRisk.HIGH: 2,
    }

    if risk_rank[response.risk] < risk_rank[allowed_risk]:
        return AIResponse(
            case_id=response.case_id,
            action=response.action,
            risk=allowed_risk,
            confidence=response.confidence,
            explanation=(
                "AI recommendation reduced the deterministic risk "
                "level. DriveMind retained the safer risk level."
            ),
        )

    return response