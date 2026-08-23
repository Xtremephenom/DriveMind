from __future__ import annotations

import random

from backend.models.system import (
    AICase,
    DecisionContext,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.dataset.scenarios import SCENARIOS


def generate_scenario_cases(
    count: int,
    *,
    seed: int = 42,
) -> list[AICase]:
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)

    cases: list[AICase] = []

    
    scenarios = list(SCENARIOS)
    rng.shuffle(scenarios)

    for index in range(count):
        scenario = scenarios[index % len(scenarios)]

        # Add controlled variation while preserving
        # the important characteristics of the scenario.
        size_multiplier = rng.choice(
            [0.5, 1.0, 1.5, 2.0]
        )

        age = scenario.age_days

        if age is not None:
            age_variation = rng.choice(
                [-2, -1, 0, 1, 2]
            )
            age = max(0, age + age_variation)

        path = scenario.path.replace(
            "file.tmp",
            f"file_{index:06d}.tmp",
        ).replace(
            "file.log",
            f"file_{index:06d}.log",
        ).replace(
            "file.dmp",
            f"file_{index:06d}.dmp",
        ).replace(
            "memory.dmp",
            f"memory_{index:06d}.dmp",
        ).replace(
            "system.log",
            f"system_{index:06d}.log",
        ).replace(
            "data.bin",
            f"data_{index:06d}.bin",
        ).replace(
            "file.pdf",
            f"file_{index:06d}.pdf",
        ).replace(
            "photo.jpg",
            f"photo_{index:06d}.jpg",
        ).replace(
            "unknown.xyz",
            f"unknown_{index:06d}.xyz",
        )

        context = DecisionContext(
            path=path,
            size=max(
                1,
                int(scenario.size * size_multiplier),
            ),
            extension=scenario.extension,
            category=scenario.category,
            exists=scenario.exists,
            age_days=age,
            is_system_path=scenario.is_system_path,
            is_user_path=scenario.is_user_path,
            is_application_path=scenario.is_application_path,
            is_locked=scenario.is_locked,
            signals=[],
            current_action=RecommendationAction.KEEP,
            current_risk=RecommendationRisk.HIGH,
        )

        cases.append(
            AICase(
                case_id=f"scenario-{index:06d}",
                context=context,
            )
        )

    return cases