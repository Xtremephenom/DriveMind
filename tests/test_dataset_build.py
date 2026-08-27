"""
Tests for the dataset build's integrity gate.

Each check gets two tests: one showing it passes on real generator output,
and one showing it *fails* on a crafted violation. The second is the one
that matters. An assertion nobody has ever seen fail is indistinguishable
from an assertion that cannot fail, and the properties guarded here --
category coverage, distinct contexts, no duplicate ids, no cross-split
leakage, no label leakage -- are exactly the ones that went unnoticed in
the previous corpus for as long as nobody counted (§589).
"""

import json
from dataclasses import replace

import pytest

from backend.models.system import AICase, FileCategory
from backend.services.ai_response import parse_ai_response
from backend.services.dataset.build import (
    DatasetIntegrityError,
    build_dataset,
    check_corpus,
    check_disjoint,
    check_no_label_leakage,
)
from backend.services.dataset.generator import generate_cases
from backend.services.evidence import SIGNAL_VOCABULARY


# Large enough to reach all ten categories and all six signals; small
# enough that the whole module stays fast.
SAMPLE = 900


@pytest.fixture(scope="module")
def cases():
    return generate_cases(SAMPLE, seed=42)


# --- check_corpus ------------------------------------------------------


def test_real_generator_output_passes_every_check(cases):
    check_corpus(cases)
    check_no_label_leakage(cases)


def test_empty_corpus_is_rejected():
    with pytest.raises(DatasetIntegrityError, match="empty"):
        check_corpus([])


def test_duplicate_case_ids_are_rejected(cases):
    with pytest.raises(DatasetIntegrityError, match="duplicate case_id"):
        check_corpus(list(cases) + [cases[0]])


def test_padded_corpus_is_rejected(cases):
    """
    Two rows with the same evidence but different ids. This is the shape
    of the defect that hid 512 distinct contexts inside 10,000 rows, so
    it is checked on the evidence rather than on the id.
    """

    clone = AICase(case_id="0000000000000000", context=cases[0].context)

    with pytest.raises(
        DatasetIntegrityError, match="distinct semantic contexts"
    ):
        check_corpus(list(cases) + [clone])


def test_missing_category_is_rejected(cases):
    """
    `cache`, `installer`, `driver`, and `system_data` were absent from the
    entire previous corpus. Nothing failed.
    """

    without_cache = [
        case
        for case in cases
        if case.context.category != FileCategory.CACHE
    ]

    with pytest.raises(DatasetIntegrityError, match="cache"):
        check_corpus(without_cache)


def test_invented_signal_is_rejected(cases):
    tampered = replace(
        cases[0].context,
        signals=["Looks deletable to me."],
    )

    with pytest.raises(
        DatasetIntegrityError, match="production cannot emit"
    ):
        check_corpus(
            [AICase(case_id=cases[0].case_id, context=tampered)]
            + list(cases[1:])
        )


def test_corpus_exercises_the_whole_signal_vocabulary(cases):
    observed = {
        signal
        for case in cases
        for signal in case.context.signals
    }

    assert observed == set(SIGNAL_VOCABULARY)


# --- check_no_label_leakage -------------------------------------------


def test_label_leakage_is_rejected(monkeypatch, cases):
    """
    Checked against a deliberately leaky serializer, because the honest
    serializer cannot produce the failure and an assertion that can never
    fail is not an assertion.
    """

    import backend.services.dataset.build as build_module

    monkeypatch.setattr(
        build_module,
        "build_ai_prompt",
        lambda case: json.dumps(
            {"current_action": case.context.current_action.value}
        ),
    )

    with pytest.raises(DatasetIntegrityError, match="current_action"):
        check_no_label_leakage(cases[:1])


# --- check_disjoint ---------------------------------------------------


def test_shared_case_id_between_sets_is_rejected(cases):
    with pytest.raises(DatasetIntegrityError, match="case_id"):
        check_disjoint(
            {
                "train": list(cases[:10]),
                "gold": list(cases[9:12]),
            }
        )


def test_shared_prompt_between_sets_is_rejected(cases):
    """
    Same evidence under a different id. The id check misses it; the
    prompt check is what catches a model being evaluated on something it
    trained on.
    """

    disguised = AICase(
        case_id="ffffffffffffffff",
        context=cases[0].context,
    )

    with pytest.raises(DatasetIntegrityError, match="prompt"):
        check_disjoint(
            {
                "train": list(cases[:10]),
                "gold": [disguised],
            }
        )


def test_disjoint_sets_pass(cases):
    check_disjoint(
        {
            "train": list(cases[:10]),
            "gold": list(cases[10:20]),
        }
    )


# --- the whole build --------------------------------------------------


def test_build_writes_five_files_and_reports_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    summary = build_dataset(SAMPLE, seed=7, gold_per_category=5)

    assert summary["total"] == SAMPLE
    assert summary["distinct_contexts"] == SAMPLE
    assert summary["gold"] == 5 * len(FileCategory)
    assert summary["red_team"] == 26
    assert set(summary["categories"]) == {
        category.value for category in FileCategory
    }

    # The 80/10/10 split, and every case landing in exactly one part of it.
    assert summary["train"] == int(SAMPLE * 0.8)
    assert summary["validation"] == int(SAMPLE * 0.1)
    assert (
        summary["train"] + summary["validation"] + summary["test"]
        == SAMPLE
    )

    assert sum(summary["actions"].values()) == SAMPLE
    assert sum(summary["risks"].values()) == SAMPLE
    assert sum(summary["categories"].values()) == SAMPLE

    data = tmp_path / "data"

    for name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "gold.jsonl",
        "red_team.jsonl",
    ):
        assert (data / name).exists(), name

    written = sum(
        1
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl")
        for _ in (data / name).read_text(encoding="utf-8").splitlines()
    )

    assert written == SAMPLE


def test_every_written_response_parses(tmp_path, monkeypatch):
    """
    The contract between the writer and the parser, asserted on the
    bytes actually written. Its absence is what let `confidence` become
    mandatory in the parser while absent from all 10,000 rows.
    """

    monkeypatch.chdir(tmp_path)

    build_dataset(200, seed=7, gold_per_category=1)

    for name in ("train.jsonl", "gold.jsonl", "red_team.jsonl"):
        lines = (
            (tmp_path / "data" / name)
            .read_text(encoding="utf-8")
            .splitlines()
        )

        assert lines

        for line in lines:
            row = json.loads(line)

            parsed = parse_ai_response(json.dumps(row["response"]))

            assert parsed.action.value == row["response"]["action"]
            assert parsed.risk.value == row["response"]["risk"]


def test_build_refuses_to_exceed_the_evidence_space(
    tmp_path, monkeypatch
):
    """
    The old generator padded silently. This one raises, and nothing is
    written when it does.
    """

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="exceeds"):
        build_dataset(1_000_000, seed=7)

    assert not (tmp_path / "data").exists()
