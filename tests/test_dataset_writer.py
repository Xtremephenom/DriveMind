import json

from backend.services.dataset.generator import generate_cases
from backend.services.dataset.writer import (
    case_to_training_example,
    write_jsonl,
)
from backend.services.decision.engine import POLICY_VERSION


def test_case_to_training_example():
    cases = generate_cases(
        1,
        seed=42,
    )

    example = case_to_training_example(cases[0])

    assert "case_id" in example
    assert "prompt" in example
    assert "response" in example

    assert "action" in example["response"]
    assert "risk" in example["response"]
    assert "explanation" in example["response"]

    assert example["response"]["action"] in {
        "keep",
        "review",
        "delete",
    }

    assert example["response"]["risk"] in {
        "low",
        "medium",
        "high",
    }


def test_write_jsonl(tmp_path):
    cases = generate_cases(
        10,
        seed=42,
    )

    output = tmp_path / "dataset.jsonl"

    count = write_jsonl(
        cases,
        output,
    )

    assert count == 10
    assert output.exists()

    lines = output.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 10

    for line in lines:
        data = json.loads(line)

        assert "case_id" in data
        assert "prompt" in data
        assert "response" in data


# --- Provenance (§514/§516) -------------------------------------------


def test_every_row_records_the_policy_that_labelled_it():
    """
    A row that travels without its policy version is an uninterpretable
    label. Asserted per row rather than once per file because the file is
    what gets uploaded, and split apart, and resampled.
    """

    cases = generate_cases(5, seed=42)

    for case in cases:
        example = case_to_training_example(case)

        assert example["policy_version"] == POLICY_VERSION


def test_the_policy_version_is_not_part_of_the_response_contract():
    """
    The model is asked for exactly `{action, risk, explanation}`. Putting
    provenance inside `response` would make the training target disagree
    with the prompt -- the shape of the defect that made every
    contract-conformant answer unparseable once already.
    """

    example = case_to_training_example(generate_cases(1, seed=42)[0])

    assert set(example["response"]) == {"action", "risk", "explanation"}
    assert "policy_version" not in example["response"]
    assert "policy_version" not in example["prompt"]
