"""
Tests for the baseline runner's arithmetic and its honesty properties.
"""

from backend.models.system import (
    AICase,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.ai_baseline import (
    format_report,
    provider_as_text,
    response_to_contract_json,
    run_baseline,
    summarize,
)
from backend.services.ai_dataset import build_baseline_cases
from backend.services.ai_response import parse_ai_response
from backend.services.ai_rule_based import RuleBasedAIProvider


def always(action: str, risk: str, explanation="Because."):
    def emit(case: AICase) -> str:
        return response_to_contract_json(
            parse_ai_response(
                '{"action": "%s", "risk": "%s", "explanation": "%s"}'
                % (action, risk, explanation)
            )
        )

    return emit


def test_contract_json_round_trips_through_the_parser():
    provider = RuleBasedAIProvider()
    case = build_baseline_cases()[0]

    text = response_to_contract_json(provider.analyze(case))
    parsed = parse_ai_response(text)

    assert parsed.action == RecommendationAction.REVIEW
    assert parsed.risk == RecommendationRisk.LOW
    assert parsed.explanation


def test_the_oracle_provider_agrees_on_every_case_by_construction():
    """
    Recorded so the number is never mistaken for a model result: the
    rule-based provider mirrors the engine, so a perfect score here says
    the harness works and nothing at all about any AI.
    """

    evaluations, summary = run_baseline(
        provider_as_text(RuleBasedAIProvider())
    )

    assert summary.cases == 12
    assert summary.structured_output_validity == 1.0
    assert summary.action_agreement == 1.0
    assert summary.risk_agreement == 1.0
    assert summary.unsafe_escalation_rate == 0.0
    assert all(e.parsed_successfully for e in evaluations)


def test_unparseable_output_scores_zero_and_is_not_hidden():
    def garbage(case: AICase) -> str:
        return "I think you should delete it."

    evaluations, summary = run_baseline(garbage)

    assert summary.structured_output_validity == 0.0
    assert summary.action_agreement == 0.0
    assert summary.risk_agreement == 0.0

    # An unparseable answer is not scored as an escalation, but it is
    # not scored as agreement either -- the denominator stays 12.
    assert summary.unsafe_escalation_rate == 0.0
    assert summary.cases == 12
    assert all(e.error for e in evaluations)


def test_a_delete_everything_model_escalates_on_every_case():
    _, summary = run_baseline(always("delete", "low"))

    assert summary.structured_output_validity == 1.0
    assert summary.action_agreement == 0.0
    assert summary.unsafe_escalation_rate == 1.0


def test_a_keep_everything_model_never_escalates():
    _, summary = run_baseline(always("keep", "high"))

    assert summary.unsafe_escalation_rate == 0.0

    # 5 of the 12 baseline cases are genuinely keep/high.
    assert summary.action_agreements == 5
    assert summary.risk_agreements == 7


def test_summary_of_nothing_does_not_divide_by_zero():
    summary = summarize([])

    assert summary.cases == 0
    assert summary.structured_output_validity == 0.0
    assert summary.unsafe_escalation_rate == 0.0


def test_report_names_every_case_and_flags_escalations():
    evaluations, summary = run_baseline(always("delete", "low"))
    report = format_report(evaluations, summary)

    for case in build_baseline_cases():
        assert case.case_id in report

    assert report.count("ESCALATION") == 12
