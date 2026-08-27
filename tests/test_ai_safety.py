"""
Safety-gate tests.

Every case here is built from *evidence*. The gate's ceiling is whatever
the deterministic engine derives from that evidence — never what the case
claims in `current_action` / `current_risk`. Two of these tests used to
pass by asserting a ceiling the engine never produced, which is precisely
the defect this file now guards against.

Engine verdicts relied on below (`policy-v1`):

    USER_DATA                      -> KEEP   / HIGH
    is_system_path (+ TEMPORARY)   -> REVIEW / HIGH
    TEMPORARY, exists, age >= 30   -> REVIEW / LOW
    TEMPORARY, exists, age <  30   -> REVIEW / MEDIUM
"""

from backend.models.system import (
    AICase,
    AIResponse,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.decision.engine import recommend_for_context
from backend.services.ai_safety.validator import (
    validate_ai_response,
)


def make_case(
    *,
    category=FileCategory.TEMPORARY,
    age_days=90,
    exists=True,
    is_system_path=False,
    is_user_path=False,
    claimed_action=RecommendationAction.KEEP,
    claimed_risk=RecommendationRisk.HIGH,
):
    """
    Build a case. `claimed_*` are the caller-supplied ground-truth
    fields; the gate must ignore them entirely.
    """

    return AICase(
        case_id="test-case-001",
        context=DecisionContext(
            path=r"C:\Temp\old.tmp",
            size=1000,
            extension=".tmp",
            category=category,
            exists=exists,
            age_days=age_days,
            is_system_path=is_system_path,
            is_user_path=is_user_path,
            is_application_path=False,
            is_locked=False,
            signals=[],
            current_action=claimed_action,
            current_risk=claimed_risk,
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
        explanation="AI recommendation.",
    )


# --- The fixtures produce the verdicts the tests below assume ----------


def test_fixture_verdicts_are_what_the_engine_actually_says():
    old_temp = recommend_for_context(make_case().context)

    assert old_temp.action == RecommendationAction.REVIEW
    assert old_temp.risk == RecommendationRisk.LOW

    user_data = recommend_for_context(
        make_case(category=FileCategory.USER_DATA).context
    )

    assert user_data.action == RecommendationAction.KEEP
    assert user_data.risk == RecommendationRisk.HIGH

    system = recommend_for_context(
        make_case(is_system_path=True).context
    )

    assert system.action == RecommendationAction.REVIEW
    assert system.risk == RecommendationRisk.HIGH


# --- Escalation is refused --------------------------------------------


def test_ai_cannot_escalate_review_to_delete():
    # Engine: REVIEW / LOW
    result = validate_ai_response(
        make_case(),
        make_response(
            RecommendationAction.DELETE,
            RecommendationRisk.LOW,
        ),
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
    assert "deterministic" in result.explanation


def test_ai_cannot_escalate_keep_to_delete():
    # Engine: KEEP / HIGH, because the evidence says user data.
    result = validate_ai_response(
        make_case(category=FileCategory.USER_DATA),
        make_response(
            RecommendationAction.DELETE,
            RecommendationRisk.LOW,
        ),
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH


def test_ai_cannot_escalate_keep_to_review():
    result = validate_ai_response(
        make_case(category=FileCategory.USER_DATA),
        make_response(
            RecommendationAction.REVIEW,
            RecommendationRisk.HIGH,
        ),
    )

    assert result.action == RecommendationAction.KEEP


# --- Risk downgrade is refused ----------------------------------------


def test_ai_cannot_reduce_risk():
    # Engine: REVIEW / HIGH, because the evidence says system path.
    result = validate_ai_response(
        make_case(is_system_path=True),
        make_response(
            RecommendationAction.REVIEW,
            RecommendationRisk.LOW,
        ),
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.HIGH
    assert "safer risk level" in result.explanation


def test_ai_may_raise_risk():
    # More conservative than the engine is always allowed through.
    result = validate_ai_response(
        make_case(),
        make_response(
            RecommendationAction.REVIEW,
            RecommendationRisk.HIGH,
        ),
    )

    assert result.risk == RecommendationRisk.HIGH


def test_safe_ai_response_is_preserved():
    response = AIResponse(
        case_id="test-case-001",
        action=RecommendationAction.REVIEW,
        risk=RecommendationRisk.LOW,
        explanation="The file should be reviewed before cleanup.",
    )

    result = validate_ai_response(make_case(), response)

    assert result is response


# --- The ceiling is not an input (§443) --------------------------------


def test_tampered_ground_truth_cannot_widen_the_gate():
    """
    A case that *claims* DELETE is permitted, over evidence that says
    user data. The gate must derive KEEP/HIGH from the evidence and
    ignore the claim entirely.
    """

    case = make_case(
        category=FileCategory.USER_DATA,
        claimed_action=RecommendationAction.DELETE,
        claimed_risk=RecommendationRisk.LOW,
    )

    result = validate_ai_response(
        case,
        make_response(
            RecommendationAction.DELETE,
            RecommendationRisk.LOW,
        ),
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH


def test_tampered_ground_truth_cannot_lower_the_risk_floor():
    case = make_case(
        is_system_path=True,
        claimed_action=RecommendationAction.REVIEW,
        claimed_risk=RecommendationRisk.LOW,
    )

    result = validate_ai_response(
        case,
        make_response(
            RecommendationAction.REVIEW,
            RecommendationRisk.LOW,
        ),
    )

    assert result.risk == RecommendationRisk.HIGH


def test_claimed_ground_truth_never_changes_the_outcome():
    """
    Sweep every claimed action/risk pair over fixed evidence. The gated
    result must be identical in all nine cases.
    """

    outcomes = set()

    for action in RecommendationAction:
        for risk in RecommendationRisk:
            result = validate_ai_response(
                make_case(
                    category=FileCategory.USER_DATA,
                    claimed_action=action,
                    claimed_risk=risk,
                ),
                make_response(
                    RecommendationAction.DELETE,
                    RecommendationRisk.LOW,
                ),
            )

            outcomes.add((result.action, result.risk))

    assert outcomes == {
        (RecommendationAction.KEEP, RecommendationRisk.HIGH)
    }

