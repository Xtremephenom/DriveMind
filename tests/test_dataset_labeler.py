from backend.models.system import RecommendationAction

from backend.services.dataset.generator import generate_cases
from backend.services.dataset.labeler import label_case


def test_labeler_uses_deterministic_engine():
    cases = generate_cases(
        20,
        seed=42,
    )

    for case in cases:
        recommendation = label_case(case)

        assert recommendation.path == case.context.path
        assert recommendation.size == case.context.size
        assert recommendation.category == case.context.category
        assert isinstance(
            recommendation.action,
            RecommendationAction,
        )