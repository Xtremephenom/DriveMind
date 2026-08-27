"""
DriveMind's deterministic safety gate.

The AI is advisory. This module is where that is enforced: it may only
ever make a recommendation *more* conservative, never less.

The ceiling is re-derived from the evidence by the deterministic engine
on every call. It is deliberately NOT read from
`case.context.current_action` / `current_risk`, because those are supplied
by whoever constructed the case — a gate whose limit is an input is not a
gate (§443 safety-gate immutability).
"""

from __future__ import annotations

from backend.models.system import (
    AICase,
    AIResponse,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.decision.engine import recommend_for_context


_ACTION_RANK = {
    RecommendationAction.KEEP: 0,
    RecommendationAction.REVIEW: 1,
    RecommendationAction.DELETE: 2,
}

_RISK_RANK = {
    RecommendationRisk.LOW: 0,
    RecommendationRisk.MEDIUM: 1,
    RecommendationRisk.HIGH: 2,
}


def is_action_escalation(
    candidate: RecommendationAction,
    ceiling: RecommendationAction,
) -> bool:
    """
    Is `candidate` more permissive than `ceiling`?

    This ordering is what the gate enforces, so anything that *reports*
    on escalation — the evaluator's unsafe-escalation metric — must ask
    here rather than keep a second copy of it (§272).
    """

    return _ACTION_RANK[candidate] > _ACTION_RANK[ceiling]


def is_risk_downgrade(
    candidate: RecommendationRisk,
    ceiling: RecommendationRisk,
) -> bool:
    """Is `candidate` less cautious than `ceiling`?"""

    return _RISK_RANK[candidate] < _RISK_RANK[ceiling]


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

    authoritative = recommend_for_context(case.context)

    allowed_action = authoritative.action
    allowed_risk = authoritative.risk

    # Never allow the AI to escalate the action.
    if is_action_escalation(response.action, allowed_action):
        return AIResponse(
            case_id=response.case_id,
            action=allowed_action,
            risk=allowed_risk,
            explanation=(
                "AI recommendation was more permissive than "
                "DriveMind's deterministic safety policy. "
                "The deterministic recommendation was enforced."
            ),
        )

    # Never allow the AI to downgrade the deterministic risk.
    if is_risk_downgrade(response.risk, allowed_risk):
        return AIResponse(
            case_id=response.case_id,
            action=response.action,
            risk=allowed_risk,
            explanation=(
                "AI recommendation reduced the deterministic risk "
                "level. DriveMind retained the safer risk level."
            ),
        )

    return response
