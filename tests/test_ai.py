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
    category=FileCategory.TEMPORARY,
    claimed_action=RecommendationAction.REVIEW,
    claimed_risk=RecommendationRisk.LOW,
):
    """
    Build a case. `claimed_*` are the ground-truth fields a caller
    supplies; the provider must derive its own verdict instead.
    """

    context = DecisionContext(
        path=r"C:\Temp\old.tmp",
        size=5000,
        extension=".tmp",
        category=category,
        exists=True,
        age_days=90,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        is_locked=False,
        signals=["Old temporary file."],
        current_action=claimed_action,
        current_risk=claimed_risk,
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
    assert result.explanation


def test_rule_based_provider_respects_keep():
    provider = RuleBasedAIProvider()

    # The evidence — not a claimed label — is what makes this KEEP/HIGH.
    result = provider.analyze(
        make_case(category=FileCategory.USER_DATA)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH
    assert result.explanation


def test_rule_based_provider_ignores_claimed_ground_truth():
    """
    The stand-in derives its verdict, so a tampered case does not move
    it. If it trusted `current_action` it would happily agree with a
    case that had been edited to permit deletion.
    """

    provider = RuleBasedAIProvider()

    tampered = provider.analyze(
        make_case(
            category=FileCategory.USER_DATA,
            claimed_action=RecommendationAction.DELETE,
            claimed_risk=RecommendationRisk.LOW,
        )
    )

    assert tampered.action == RecommendationAction.KEEP
    assert tampered.risk == RecommendationRisk.HIGH
