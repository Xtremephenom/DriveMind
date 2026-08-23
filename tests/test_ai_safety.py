from backend.models.system import (
    AICase,
    AIResponse,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_safety.validator import (
    validate_ai_response,
)


def make_case(
    action=RecommendationAction.REVIEW,
    risk=RecommendationRisk.LOW,
):
    return AICase(
        case_id="test-case-001",
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


def make_response(
    action=RecommendationAction.DELETE,
    risk=RecommendationRisk.LOW,
):
    return AIResponse(
        case_id="test-case-001",
        action=action,
        risk=risk,
        confidence=0.95,
        explanation="AI recommendation.",
    )


def test_ai_cannot_escalate_review_to_delete():
    case = make_case(
        RecommendationAction.REVIEW,
        RecommendationRisk.LOW,
    )

    response = make_response(
        RecommendationAction.DELETE,
        RecommendationRisk.LOW,
    )

    result = validate_ai_response(case, response)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW


def test_ai_cannot_escalate_keep_to_delete():
    case = make_case(
        RecommendationAction.KEEP,
        RecommendationRisk.HIGH,
    )

    response = make_response(
        RecommendationAction.DELETE,
        RecommendationRisk.LOW,
    )

    result = validate_ai_response(case, response)

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH


def test_ai_cannot_reduce_risk():
    case = make_case(
        RecommendationAction.REVIEW,
        RecommendationRisk.HIGH,
    )

    response = make_response(
        RecommendationAction.REVIEW,
        RecommendationRisk.LOW,
    )

    result = validate_ai_response(case, response)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.HIGH


def test_safe_ai_response_is_preserved():
    case = make_case(
        RecommendationAction.REVIEW,
        RecommendationRisk.LOW,
    )

    response = AIResponse(
        case_id="test-case-001",
        action=RecommendationAction.REVIEW,
        risk=RecommendationRisk.LOW,
        confidence=0.91,
        explanation="The file should be reviewed before cleanup.",
    )

    result = validate_ai_response(case, response)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
    assert result.confidence == 0.91