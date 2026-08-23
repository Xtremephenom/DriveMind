from backend.services.dataset.generator import generate_cases
from backend.services.dataset.split import split_cases


def test_split_sizes():
    cases = generate_cases(
        100,
        seed=42,
    )

    result = split_cases(
        cases,
        seed=42,
    )

    assert len(result.train) == 80
    assert len(result.validation) == 10
    assert len(result.test) == 10


def test_split_has_no_overlap():
    cases = generate_cases(
        100,
        seed=42,
    )

    result = split_cases(
        cases,
        seed=42,
    )

    train_ids = {case.case_id for case in result.train}
    validation_ids = {case.case_id for case in result.validation}
    test_ids = {case.case_id for case in result.test}

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_split_is_reproducible():
    cases = generate_cases(
        100,
        seed=42,
    )

    first = split_cases(
        cases,
        seed=42,
    )

    second = split_cases(
        cases,
        seed=42,
    )

    assert [
        case.case_id for case in first.train
    ] == [
        case.case_id for case in second.train
    ]

    assert [
        case.case_id for case in first.validation
    ] == [
        case.case_id for case in second.validation
    ]

    assert [
        case.case_id for case in first.test
    ] == [
        case.case_id for case in second.test
    ]