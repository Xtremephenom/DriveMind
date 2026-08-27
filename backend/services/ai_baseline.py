"""
The AI baseline runner.

Aggregates `ai_evaluator` results over the baseline cases into the four
numbers §55/§56/§58 ask for: structured-output validity, action
agreement, risk agreement, and — the primary safety metric — unsafe
escalation rate.

Two conventions, both deliberate and both stated in the report:

*   Agreement is measured over **all** cases, not over the ones that
    parsed. A model that emits garbage has not agreed with anything, and
    hiding those cases in the denominator would flatter it.
*   Unsafe escalation counts every answer more permissive than the
    deterministic verdict, including the ones the safety gate goes on to
    block. The gate stopping an escalation does not make the model's
    answer safe; it makes the *product* safe.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from backend.models.system import AICase, AIResponse
from backend.services.ai import AIProvider
from backend.services.ai_dataset import build_baseline_cases
from backend.services.ai_evaluator import (
    AIEvaluation,
    evaluate_ai_response,
)
from backend.services.ai_rule_based import RuleBasedAIProvider


def response_to_contract_json(response: AIResponse) -> str:
    """
    Serialize a response into exactly the JSON the prompt asks for.

    Used to feed a structured provider through the text-based evaluator,
    so the production parser is on the baseline path too.
    """

    return json.dumps(
        {
            "action": response.action.value,
            "risk": response.risk.value,
            "explanation": response.explanation,
        },
        ensure_ascii=False,
    )


def provider_as_text(
    provider: AIProvider,
) -> Callable[[AICase], str]:
    """Adapt an `AIProvider` to the raw-text interface the evaluator takes."""

    def emit(case: AICase) -> str:
        return response_to_contract_json(provider.analyze(case))

    return emit


@dataclass(frozen=True)
class BaselineSummary:
    cases: int
    parsed: int
    action_agreements: int
    risk_agreements: int
    unsafe_escalations: int

    def _rate(self, count: int) -> float:
        if self.cases == 0:
            return 0.0

        return count / self.cases

    @property
    def structured_output_validity(self) -> float:
        return self._rate(self.parsed)

    @property
    def action_agreement(self) -> float:
        return self._rate(self.action_agreements)

    @property
    def risk_agreement(self) -> float:
        return self._rate(self.risk_agreements)

    @property
    def unsafe_escalation_rate(self) -> float:
        return self._rate(self.unsafe_escalations)


def summarize(
    evaluations: list[AIEvaluation],
) -> BaselineSummary:
    return BaselineSummary(
        cases=len(evaluations),
        parsed=sum(1 for e in evaluations if e.parsed_successfully),
        action_agreements=sum(1 for e in evaluations if e.action_agreement),
        risk_agreements=sum(1 for e in evaluations if e.risk_agreement),
        unsafe_escalations=sum(1 for e in evaluations if e.unsafe_escalation),
    )


def run_baseline(
    emit_text: Callable[[AICase], str],
    cases: list[AICase] | None = None,
) -> tuple[list[AIEvaluation], BaselineSummary]:
    """
    Evaluate `emit_text` over the baseline cases.

    `emit_text` receives a case and returns the model's raw text, which
    is what a local LLM actually produces. A provider that returns a
    structured `AIResponse` can be adapted with `provider_as_text`.
    """

    if cases is None:
        cases = build_baseline_cases()

    evaluations = [
        evaluate_ai_response(case, emit_text(case))
        for case in cases
    ]

    return evaluations, summarize(evaluations)


def format_report(
    evaluations: list[AIEvaluation],
    summary: BaselineSummary,
) -> str:
    lines = [
        f"cases                      {summary.cases}",
        f"structured-output validity {summary.structured_output_validity:.1%}"
        f"  ({summary.parsed}/{summary.cases})",
        f"action agreement           {summary.action_agreement:.1%}"
        f"  ({summary.action_agreements}/{summary.cases})",
        f"risk agreement             {summary.risk_agreement:.1%}"
        f"  ({summary.risk_agreements}/{summary.cases})",
        f"unsafe escalation rate     {summary.unsafe_escalation_rate:.1%}"
        f"  ({summary.unsafe_escalations}/{summary.cases})",
        "",
        "per case:",
    ]

    for evaluation in evaluations:
        lines.append(
            f"  {evaluation.case_id:<16}"
            f" expected {evaluation.expected_action.value}/"
            f"{evaluation.expected_risk.value:<8}"
            f" got "
            + (
                f"{evaluation.ai_action.value}/{evaluation.ai_risk.value}"
                if evaluation.parsed_successfully
                else f"UNPARSEABLE ({evaluation.error})"
            )
            + ("  ESCALATION" if evaluation.unsafe_escalation else "")
        )

    return "\n".join(lines)


def main() -> None:
    evaluations, summary = run_baseline(
        provider_as_text(RuleBasedAIProvider())
    )

    print(
        "provider: RuleBasedAIProvider - a deterministic mirror of the\n"
        "engine, not a model. Its agreement is 1.0 and its escalation\n"
        "rate 0.0 by construction; these numbers measure the harness,\n"
        "not any AI capability.\n"
    )
    print(format_report(evaluations, summary))


if __name__ == "__main__":
    main()
