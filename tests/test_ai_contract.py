"""
Contract tests between the training data and the runtime parser.

The AI output contract is exactly {action, risk, explanation}.

Three places must agree on it: the prompt the model is given
(`ai_prompt.SYSTEM_INSTRUCTION`), the labels the model is trained on
(`dataset.writer.case_to_training_example`) and the parser that accepts
the model's output at runtime (`ai_response.parse_ai_response`).

They silently disagreed once already: the prompt and the dataset dropped
`confidence` while the parser still required it, which made every
conformant model response unparseable. These tests exist so that
divergence fails the suite instead of the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.ai_prompt import SYSTEM_INSTRUCTION
from backend.services.ai_response import parse_ai_response
from backend.services.dataset.generator import generate_cases
from backend.services.dataset.writer import case_to_training_example


CONTRACT_FIELDS = {"action", "risk", "explanation"}

DATASET_FILES = (
    "data/train.jsonl",
    "data/validation.jsonl",
    "data/test.jsonl",
)


def test_training_response_shape_is_exactly_the_contract():
    for case in generate_cases(50, seed=7):
        example = case_to_training_example(case)

        assert set(example["response"].keys()) == CONTRACT_FIELDS


def test_parser_accepts_every_generated_training_label():
    for case in generate_cases(200, seed=7):
        example = case_to_training_example(case)

        raw = json.dumps(example["response"])

        parsed = parse_ai_response(raw)

        assert parsed.action.value == example["response"]["action"]
        assert parsed.risk.value == example["response"]["risk"]
        assert parsed.explanation == example["response"]["explanation"]


@pytest.mark.parametrize("dataset_file", DATASET_FILES)
def test_parser_accepts_rows_from_the_written_dataset(dataset_file):
    path = Path(dataset_file)

    if not path.exists():
        pytest.skip(f"{dataset_file} has not been built")

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)

            response = row["response"]

            assert set(response.keys()) == CONTRACT_FIELDS, (
                f"{dataset_file}:{line_number} response fields "
                f"{sorted(response.keys())} do not match the contract"
            )

            parsed = parse_ai_response(json.dumps(response))

            assert parsed.action.value == response["action"]
            assert parsed.risk.value == response["risk"]


def test_prompt_and_parser_agree_on_the_contract():
    # Every contract field the prompt asks for must be a field the
    # parser requires, and vice versa.
    for field in CONTRACT_FIELDS:
        assert f'"{field}"' in SYSTEM_INSTRUCTION

    minimal = {field: "" for field in CONTRACT_FIELDS}

    for field in sorted(CONTRACT_FIELDS):
        incomplete = {
            key: value
            for key, value in minimal.items()
            if key != field
        }

        with pytest.raises(
            ValueError,
            match="missing required fields",
        ):
            parse_ai_response(json.dumps(incomplete))

