from backend.models.system import (
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_cases import (
    build_ai_case,
    build_case_id,
)


def make_context():
    return DecisionContext(
        path=r"C:\Temp\old.tmp",
        size=5000,
        extension=".tmp",
        category=FileCategory.TEMPORARY,
        exists=True,
        age_days=90,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        is_locked=False,
        signals=[
            "Temporary file.",
            "Older than 30 days.",
        ],
        current_action=RecommendationAction.REVIEW,
        current_risk=RecommendationRisk.LOW,
    )


def test_case_id_is_deterministic():
    context = make_context()

    first = build_case_id(context)
    second = build_case_id(context)

    assert first == second
    assert len(first) == 16


def test_case_id_does_not_expose_path():
    context = make_context()

    case_id = build_case_id(context)

    assert context.path not in case_id


def test_build_ai_case():
    context = make_context()

    case = build_ai_case(context)

    assert case.context is context
    assert case.case_id == build_case_id(context)