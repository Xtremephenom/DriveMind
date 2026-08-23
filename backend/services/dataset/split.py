from __future__ import annotations

import random
from dataclasses import dataclass

from backend.models.system import AICase


@dataclass(frozen=True)
class DatasetSplit:
    train: list[AICase]
    validation: list[AICase]
    test: list[AICase]


def split_cases(
    cases: list[AICase],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    if not cases:
        raise ValueError("cases must not be empty")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1"
        )

    shuffled = list(cases)

    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)

    train_end = int(total * train_ratio)
    validation_end = train_end + int(
        total * validation_ratio
    )

    return DatasetSplit(
        train=shuffled[:train_end],
        validation=shuffled[train_end:validation_end],
        test=shuffled[validation_end:],
    )