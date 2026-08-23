from __future__ import annotations

from collections import Counter

from backend.services.dataset.labeler import label_case
from backend.services.dataset.scenario_generator import (
    generate_scenario_cases,
)
from backend.services.dataset.split import split_cases
from backend.services.dataset.writer import write_jsonl


def build_dataset(
    count: int = 10_000,
    *,
    seed: int = 42,
) -> dict:
    cases = generate_scenario_cases(
        count,
        seed=seed,
    )

    actions = Counter()
    risks = Counter()
    categories = Counter()

    for case in cases:
        recommendation = label_case(case)

        actions[recommendation.action.value] += 1
        risks[recommendation.risk.value] += 1
        categories[case.context.category.value] += 1

    split = split_cases(
        cases,
        seed=seed,
    )

    write_jsonl(
        split.train,
        "data/train.jsonl",
    )

    write_jsonl(
        split.validation,
        "data/validation.jsonl",
    )

    write_jsonl(
        split.test,
        "data/test.jsonl",
    )

    return {
        "total": len(cases),
        "actions": dict(actions),
        "risks": dict(risks),
        "categories": dict(categories),
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
