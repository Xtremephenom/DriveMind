"""
Guards on the red-team seed set.

The point of these is that `SCENARIOS` is hand-written, so nothing else
stops a hand-written mistake. In particular they assert that a scenario's
path flags are the ones production would derive from its path, which is
the defect this file used to have: flags were hand-set and three of them
disagreed with `build_path_evidence`.
"""

import pytest

from backend.models.system import FileCategory
from backend.services.decision.engine import recommend_for_context
from backend.services.dataset.scenarios import (
    SCENARIOS,
    Scenario,
    generate_red_team_cases,
    scenario_to_case,
)
from backend.services.evidence import SIGNAL_VOCABULARY


def test_scenarios_are_not_empty():
    assert SCENARIOS


def test_scenario_names_are_unique():
    names = [scenario.name for scenario in SCENARIOS]

    assert len(names) == len(set(names))


def test_scenarios_have_valid_sizes():
    for scenario in SCENARIOS:
        assert scenario.size > 0


def test_scenarios_cover_core_categories():
    categories = {
        scenario.category.value for scenario in SCENARIOS
    }

    assert {
        "temporary",
        "log",
        "crash_dump",
        "user_data",
        "application_data",
        "unknown",
    } <= categories


def test_scenarios_cover_conflicting_evidence():
    names = {scenario.name for scenario in SCENARIOS}

    expected = {
        "temp_old_locked",
        "temp_recent_locked",
        "temp_old_missing",
        "temp_recent_large",
        "system_old_locked_log",
        "system_recent_log",
        "system_old_dump_locked",
        "user_old_locked",
        "user_missing",
        "application_missing",
        "unknown_locked",
        "unknown_missing",
    }

    assert expected <= names


def test_a_missing_file_cannot_carry_an_age():
    """
    Production derives age from a stat call, so `exists=False` with an age
    is evidence the scanner cannot produce. The old scenario table had
    five such rows.
    """

    with pytest.raises(ValueError):
        Scenario(
            name="impossible",
            category=FileCategory.TEMPORARY,
            path=r"C:\Temp\gone.tmp",
            age_days=90,
            exists=False,
            is_locked=False,
            size=1024,
        )


# --- The cases built from them ----------------------------------------


def test_red_team_covers_every_scenario_with_unique_ids():
    cases = generate_red_team_cases()

    assert len(cases) == len(SCENARIOS)
    assert len({case.case_id for case in cases}) == len(SCENARIOS)


def test_scenario_flags_are_derived_not_asserted():
    """
    A scenario says where a file is; production decides what that means.
    `C:\\ProgramData\\TestApp\\data.bin` used to claim
    `is_application_path=True` -- the substring production matches is
    `\\program files\\`, so the correct answer is False (ADR 0002, Gap 2).
    """

    by_name = {
        scenario.name: scenario_to_case(scenario)
        for scenario in SCENARIOS
    }

    application = by_name["application_data"].context

    assert application.is_application_path is False
    assert application.is_system_path is False
    assert application.is_user_path is False

    system_log = by_name["system_log"].context

    assert system_log.is_system_path is True

    user_document = by_name["user_document"].context

    assert user_document.is_user_path is True


def test_red_team_labels_come_from_the_engine():
    for case in generate_red_team_cases():
        authoritative = recommend_for_context(case.context)

        assert case.context.current_action == authoritative.action
        assert case.context.current_risk == authoritative.risk


def test_red_team_signals_use_the_production_vocabulary():
    for case in generate_red_team_cases():
        for signal in case.context.signals:
            assert signal in SIGNAL_VOCABULARY


def test_locked_cases_survive_into_the_red_team_set():
    """
    `is_locked` is excluded from the training corpus because production
    never sets it (ADR 0002, Gap 1). The red-team set is where it lives,
    so its presence here is load-bearing, not incidental.
    """

    locked = [
        case
        for case in generate_red_team_cases()
        if case.context.is_locked
    ]

    assert len(locked) >= 6
