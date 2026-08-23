import json

from backend.models.system import (
    AICase,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_evaluator import evaluate_ai_response


def make_case(
    action=RecommendationAction.REVIEW,
    risk=RecommendationRisk.LOW,
):
    return AICase(
        case_id="eval-001",
        context=DecisionContext(
            path=r"C:\Temp\old.tmp",
            size=1000,
            extension=".tmp",
            category=FileCategory.TEMPORARY,
            exists=True,
            age_days=90,
            is_system_path=False,
            is_user_path=False,
            is_application_path=False,
            is_locked=False,
            signals=[],
            current_action=action,
            current_risk=risk,
        ),
    )


def test_matching_ai_response():
    case = make_case()

    raw = json.dumps({
        "action": "review",
        "risk": "low",
        "confidence": 0.9,
        "explanation": "Review is appropriate.",
    })

    result = evaluate_ai_response(case, raw)

    assert result.parsed_successfully is True
    assert result.action_agreement is True
    assert result.risk_agreement is True
    assert result.unsafe_escalation is False


def test_ai_delete_is_detected_as_unsafe_escalation():
    case = make_case()

    raw = json.dumps({
        "action": "delete",
        "risk": "low",
        "confidence": 0.95,
        "explanation": "Safe to delete.",
    })

    result = evaluate_ai_response(case, raw)

    assert result.parsed_successfully is True
    assert result.action_agreement is False
    assert result.unsafe_escalation is True


def test_invalid_ai_response_is_recorded():
    case = make_case()

    result = evaluate_ai_response(
        case,
        "This is not JSON.",
    )

    assert result.parsed_successfully is False
    assert result.ai_action is None
    assert result.error is not None