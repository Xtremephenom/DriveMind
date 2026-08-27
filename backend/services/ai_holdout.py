"""
Held-out evaluation, split by split.

`ai_baseline` runs a provider over the 12 hand-built cases in
`ai_dataset.build_baseline_cases()`. That is a smoke test for the harness,
not a measurement of a model: 12 cases cannot distinguish a competent model
from a lucky one. [`docs/model-card.md`](../../docs/model-card.md) fixes the
real bar in advance -- all four metrics on `data/test.jsonl`, with
`data/gold.jsonl` and `data/red_team.jsonl` reported **separately**, because
averaging adversarial performance into an overall figure hides exactly what
the red-team set exists to expose.

This module is what produces those numbers. It deliberately does two things
the obvious implementation would not:

*   **It evaluates real `AICase` objects, regenerated from the seed, not
    the `prompt` strings in the JSONL.** The evaluator has to re-derive
    ground truth by running the engine on the case's evidence (§65), and a
    prompt string has had the evidence flattened out of it. Reading the
    row's `response` as ground truth instead would measure whoever built
    the corpus.
*   **It then proves the regenerated cases are the published ones**, by
    matching `case_id` and the rendered prompt against the written files,
    in order, before any metric is computed. Without that check "evaluated
    on the held-out set" is an assumption; with it, a generator change that
    silently moves the split boundary is a hard failure instead of a
    quietly different number.

Nothing here writes to disk. The report goes to stdout.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.models.system import AICase
from backend.services.ai import AIProvider
from backend.services.ai_baseline import (
    BaselineSummary,
    provider_as_text,
    summarize,
)
from backend.services.ai_evaluator import AIEvaluation, evaluate_ai_response
from backend.services.ai_prompt import build_ai_prompt
from backend.services.ai_rule_based import RuleBasedAIProvider
from backend.services.dataset.generator import generate_corpus_and_gold
from backend.services.dataset.scenarios import generate_red_team_cases
from backend.services.dataset.split import split_cases
from backend.services.decision.engine import POLICY_VERSION

DATA_DIR = Path("data")

# `train` and `validation` are deliberately absent. Training data is not an
# evaluation set, and a harness that can be pointed at it is a harness that
# will eventually be pointed at it.
HELD_OUT = ("test", "gold", "red_team")


class HoldoutMismatch(AssertionError):
    """
    The regenerated cases are not the ones in `data/`.

    Raised instead of evaluating, because the alternative is a number
    labelled "held-out test set" that was measured on something else.
    """


def load_held_out_cases(
    count: int = 10_000,
    *,
    seed: int = 42,
    gold_per_category: int = 10,
) -> dict[str, list[AICase]]:
    """
    Regenerate the three held-out sets, in the order they were written.

    Same calls, same arguments, and same seed as `build.build_dataset`, so
    the sets are the published ones by construction -- and then checked
    rather than trusted by `verify_against_written`.
    """

    corpus, gold = generate_corpus_and_gold(
        count,
        gold_per_category=gold_per_category,
        seed=seed,
    )

    split = split_cases(corpus, seed=seed)

    return {
        "test": split.test,
        "gold": gold,
        "red_team": generate_red_team_cases(),
    }

def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise HoldoutMismatch(
            f"{path} is missing. `data/` is gitignored, so a clone has no "
            "rows until the corpus is rebuilt:\n"
            "  python -c \"from backend.services.dataset.build import "
            'build_dataset; build_dataset()"'
        )

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_against_written(
    cases_by_split: dict[str, list[AICase]],
    data_dir: Path = DATA_DIR,
) -> dict[str, int]:
    """
    Assert the regenerated cases are the written rows, in order.

    Both `case_id` and the rendered prompt are compared. The id alone would
    pass if the id scheme stopped depending on some part of the evidence;
    the prompt alone would pass if two different cases rendered
    identically. Together they pin the case the model will actually see.
    """

    counts: dict[str, int] = {}

    for name, cases in cases_by_split.items():
        rows = _read_rows(data_dir / f"{name}.jsonl")

        if len(rows) != len(cases):
            raise HoldoutMismatch(
                f"{name}: {len(rows)} rows written, {len(cases)} "
                "regenerated"
            )

        for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
            if row["case_id"] != case.case_id:
                raise HoldoutMismatch(
                    f"{name}[{index}]: case_id {row['case_id']} written, "
                    f"{case.case_id} regenerated"
                )

            if row["prompt"] != build_ai_prompt(case):
                raise HoldoutMismatch(
                    f"{name}[{index}] ({case.case_id}): the written prompt "
                    "and the regenerated prompt differ"
                )

            if row["policy_version"] != POLICY_VERSION:
                raise HoldoutMismatch(
                    f"{name}[{index}] ({case.case_id}): written under "
                    f"{row['policy_version']}, engine is {POLICY_VERSION}; "
                    "metrics across policy versions are not comparable"
                )

        counts[name] = len(rows)

    return counts

@dataclass(frozen=True)
class SplitResult:
    name: str
    cases: list[AICase]
    evaluations: list[AIEvaluation]
    summary: BaselineSummary


def run_held_out(
    emit_text: Callable[[AICase], str],
    cases_by_split: dict[str, list[AICase]],
) -> list[SplitResult]:
    """
    Evaluate `emit_text` over each held-out set independently.

    Independently, and reported that way: there is no combined figure
    anywhere in this module. An overall number that mixes 1,000 ordinary
    cases with 26 adversarial ones is dominated by the ordinary ones, which
    is the arithmetic that lets a model fail every red-team probe and still
    report 97%.
    """

    results = []

    for name in HELD_OUT:
        cases = cases_by_split[name]

        evaluations = [
            evaluate_ai_response(case, emit_text(case))
            for case in cases
        ]

        results.append(
            SplitResult(
                name=name,
                cases=cases,
                evaluations=evaluations,
                summary=summarize(evaluations),
            )
        )

    return results

def _metrics(summary: BaselineSummary) -> list[str]:
    total = summary.cases

    return [
        f"    structured-output validity  {summary.structured_output_validity:>7.1%}"
        f"  ({summary.parsed}/{total})",
        f"    action agreement            {summary.action_agreement:>7.1%}"
        f"  ({summary.action_agreements}/{total})",
        f"    risk agreement              {summary.risk_agreement:>7.1%}"
        f"  ({summary.risk_agreements}/{total})",
        f"    unsafe escalation rate      {summary.unsafe_escalation_rate:>7.1%}"
        f"  ({summary.unsafe_escalations}/{total})   before the gate",
    ]


def _per_category(result: SplitResult) -> list[str]:
    """
    The stratified breakdown, for the set that is stratified.

    `gold` holds an equal number of cases per `FileCategory`; a single
    aggregate over it discards the only property that distinguishes it from
    a random sample of the same size.
    """

    buckets: dict[str, list[AIEvaluation]] = {}

    for case, evaluation in zip(result.cases, result.evaluations, strict=True):
        buckets.setdefault(
            case.context.category.value,
            [],
        ).append(evaluation)

    lines = ["    per category:"]

    for category in sorted(buckets):
        group = buckets[category]
        summary = summarize(group)

        lines.append(
            f"      {category:<18}"
            f" parsed {summary.parsed}/{summary.cases}"
            f"  action {summary.action_agreements}/{summary.cases}"
            f"  risk {summary.risk_agreements}/{summary.cases}"
            f"  escalations {summary.unsafe_escalations}"
        )

    return lines

def _failures(result: SplitResult, limit: int = 10) -> list[str]:
    """
    The cases that went wrong, named.

    A rate says how often; these say which. Escalations come first because
    an escalation is a safety event and a parse failure is not.
    """

    lines: list[str] = []

    escalations = [e for e in result.evaluations if e.unsafe_escalation]
    unparsed = [e for e in result.evaluations if not e.parsed_successfully]

    if escalations:
        lines.append(f"    escalations ({len(escalations)}):")

        for evaluation in escalations[:limit]:
            lines.append(
                f"      {evaluation.case_id}  engine "
                f"{evaluation.expected_action.value} -> model "
                f"{evaluation.ai_action.value}"
            )

        if len(escalations) > limit:
            lines.append(f"      ... {len(escalations) - limit} more")

    if unparsed:
        lines.append(f"    unparseable ({len(unparsed)}):")

        for evaluation in unparsed[:limit]:
            lines.append(f"      {evaluation.case_id}  {evaluation.error}")

        if len(unparsed) > limit:
            lines.append(f"      ... {len(unparsed) - limit} more")

    return lines

def format_held_out_report(
    results: list[SplitResult],
    provider_name: str,
    verified: dict[str, int],
) -> str:
    lines = [
        "DriveMind held-out evaluation",
        f"  provider        {provider_name}",
        # On every block, not once in a footer: a metric whose policy
        # version is unknown cannot be compared with another (§514/§516).
        f"  policy version  {POLICY_VERSION}",
        "  verified        "
        + ", ".join(
            f"data/{name}.jsonl ({count} rows)"
            for name, count in verified.items()
        ),
        "                  match the regenerated cases by case_id, prompt,",
        "                  and policy_version",
        "",
    ]

    for result in results:
        lines.append(f"  {result.name}  ({result.summary.cases} cases)")
        lines.extend(_metrics(result.summary))

        if result.name == "gold":
            lines.extend(_per_category(result))

        lines.extend(_failures(result))
        lines.append("")

    lines.append(
        "  No combined figure is reported. Averaging red_team into an "
        "overall\n  number hides what the red_team set exists to expose."
    )

    return "\n".join(lines)

def evaluate_emitter(
    emit_text: Callable[[AICase], str],
    *,
    data_dir: Path = DATA_DIR,
) -> tuple[list[SplitResult], dict[str, int]]:
    """
    Load, verify, then evaluate. Verification is not optional.

    Takes the raw-text interface rather than an `AIProvider`, because that
    is what a language model actually is: something that emits a string
    which may or may not satisfy the contract. `parse_ai_response` decides
    whether it did, and a provider that returns a well-formed `AIResponse`
    object has already had that question answered for it.
    """

    cases_by_split = load_held_out_cases()
    verified = verify_against_written(cases_by_split, data_dir)

    return run_held_out(emit_text, cases_by_split), verified


def evaluate_provider(
    provider: AIProvider,
    *,
    data_dir: Path = DATA_DIR,
) -> tuple[list[SplitResult], dict[str, int]]:
    return evaluate_emitter(
        provider_as_text(provider),
        data_dir=data_dir,
    )


def main() -> None:
    provider = RuleBasedAIProvider()

    results, verified = evaluate_provider(provider)

    print(
        "provider: RuleBasedAIProvider - a deterministic mirror of the\n"
        "engine, not a model. Every figure below is 1.0 or 0.0 by\n"
        "construction. What this run measures is that the harness scores\n"
        "1,126 held-out cases correctly, and that the cases it scored are\n"
        "the published ones. It measures nothing about a language model.\n"
    )
    print(
        format_held_out_report(
            results,
            type(provider).__name__,
            verified,
        )
    )


if __name__ == "__main__":
    main()
