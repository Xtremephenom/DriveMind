import json

from backend.services.dataset.generator import generate_cases
from backend.services.dataset.writer import (
    case_to_training_example,
    write_jsonl,
)


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