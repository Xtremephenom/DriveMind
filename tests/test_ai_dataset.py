"""
Tests for the 12-case baseline fixture.

The point of every test here is that the fixture must not be able to
carry a label the engine does not produce. The previous hand-written
table recorded `locked-temp` as KEEP/HIGH against an engine that returns
REVIEW/LOW, and nothing in the suite noticed.
"""

from backend.models.system import (
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.ai_dataset import build_baseline_cases
from backend.services.decision.engine import recommend_for_context


def test_baseline_has_the_expected_shape():
    cases = build_baseline_cases()

    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12
    assert all(case.case_id for case in cases)


def test_every_baseline_label_is_the_engine_verdict():
    for case in build_baseline_cases():
        authoritative = recommend_for_context(case.context)

        assert case.context.current_action == authoritative.action, (
            f"{case.case_id}: recorded action disagrees with the engine"
        )
        assert case.context.current_risk == authoritative.risk, (
            f"{case.case_id}: recorded risk disagrees with the engine"
        )


def test_locked_temp_carries_the_engines_verdict_not_the_intended_one():
    """
    The gap ADR 0002 records: `is_locked` is read nowhere, so a locked
    temporary file is scored exactly like an unlocked one. The fixture
    must say so rather than assert the policy we wish existed.
    """

    cases = {case.case_id: case for case in build_baseline_cases()}

    locked = cases["locked-temp"]
    unlocked = cases["temp-old"]

    assert locked.context.is_locked is True
    assert locked.context.current_action == RecommendationAction.REVIEW
    assert locked.context.current_risk == RecommendationRisk.LOW

    assert (
        locked.context.current_action == unlocked.context.current_action
    )
    assert locked.context.current_risk == unlocked.context.current_risk


def test_baseline_contains_no_delete_labels():
    """policy-v1 emits no `delete`; the baseline must reflect that."""

    actions = {
        case.context.current_action
        for case in build_baseline_cases()
    }

    assert RecommendationAction.DELETE not in actions
