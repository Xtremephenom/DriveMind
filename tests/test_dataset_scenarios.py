from backend.services.dataset.scenarios import SCENARIOS


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
        scenario.category
        for scenario in SCENARIOS
    }

    assert "temporary" in {
        category.value for category in categories
    }

    assert "log" in {
        category.value for category in categories
    }

    assert "crash_dump" in {
        category.value for category in categories
    }

    assert "user_data" in {
        category.value for category in categories
    }

    assert "application_data" in {
        category.value for category in categories
    }

    assert "unknown" in {
        category.value for category in categories
    }

def test_scenarios_cover_conflicting_evidence():
    names = {
        scenario.name
        for scenario in SCENARIOS
    }

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

    assert expected.issubset(names)