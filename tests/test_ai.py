from backend.models.system import (
    AICase,
    AIResponse,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_rule_based import (
    RuleBasedAIProvider,
)


def make_case(
    action=RecommendationAction.REVIEW,
    risk=RecommendationRisk.LOW,
):
    context = DecisionContext(
        path=r"C:\Temp\old.tmp",
        size=5000,
        extension=".tmp",
        category=FileCategory.TEMPORARY,
        exists=True,
        age_days=90,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        is_locked=False,
        signals=["Old temporary file."],
        current_action=action,
        current_risk=risk,
    )

    return AICase(
        case_id="test-case-001",
        context=context,
    )


def test_rule_based_provider_returns_ai_response():
    provider = RuleBasedAIProvider()

    result = provider.analyze(make_case())

    assert isinstance(result, AIResponse)
    assert result.case_id == "test-case-001"
    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
    assert 0.0 <= result.confidence <= 1.0
    assert result.explanation


def test_rule_based_provider_respects_keep():
    provider = RuleBasedAIProvider()

    result = provider.analyze(
        make_case(
            action=RecommendationAction.KEEP,
            risk=RecommendationRisk.HIGH,
        )
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH
    assert result.confidence >= 0.9