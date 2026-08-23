from backend.services.dataset.scenario_generator import (
    generate_scenario_cases,
)


def test_scenario_generator_count():
    cases = generate_scenario_cases(
        100,
        seed=42,
    )

    assert len(cases) == 100


def test_scenario_generator_ids_are_unique():
    cases = generate_scenario_cases(
        100,
        seed=42,
    )

    ids = [case.case_id for case in cases]

    assert len(ids) == len(set(ids))


def test_scenario_generator_is_reproducible():
    first = generate_scenario_cases(
        100,
        seed=42,
    )

    second = generate_scenario_cases(
        100,
        seed=42,
    )

    assert first == second


def test_scenario_generator_covers_multiple_categories():
    cases = generate_scenario_cases(
        500,
        seed=42,
    )

    categories = {
        case.context.category
        for case in cases
    }

    assert len(categories) >= 5