from __future__ import annotations

import json
from pathlib import Path

from backend.models.system import AICase
from backend.services.ai_prompt import build_ai_prompt
from backend.services.dataset.labeler import label_case


def case_to_training_example(case: AICase) -> dict:
    recommendation = label_case(case)

    return {
        "case_id": case.case_id,
        "prompt": build_ai_prompt(case),
        "response": {
            "action": recommendation.action.value,
            "risk": recommendation.risk.value,
            "explanation": recommendation.reason,
        },
    }


def write_jsonl(
    cases: list[AICase],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for case in cases:
            example = case_to_training_example(case)

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return len(cases)