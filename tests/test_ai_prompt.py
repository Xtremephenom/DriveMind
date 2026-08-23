from backend.models.system import (
    AICase,
    DecisionContext,
    FileCategory,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_prompt import (
    SYSTEM_INSTRUCTION,
    build_ai_prompt,
)


def make_case():
    context = DecisionContext(
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

    return AICase(
        case_id="test-case-001",
        context=context,
    )


def test_prompt_contains_case_data():
    prompt = build_ai_prompt(make_case())

    assert "test-case-001" in prompt
    assert r"C:\\Temp\\old.tmp" in prompt
    assert ".tmp" in prompt
    assert "temporary" in prompt
    assert "90" in prompt


def test_prompt_contains_safety_instruction():
    prompt = build_ai_prompt(make_case())

    assert "Never assume a file is safe to delete" in prompt
    assert "Never invent facts" in prompt
    assert "Never execute commands" in prompt


def test_prompt_requires_json():
    prompt = build_ai_prompt(make_case())

    assert "valid JSON" in prompt
    assert '"action"' in prompt
    assert '"risk"' in prompt
    assert '"confidence"' in prompt
    assert '"explanation"' in prompt


def test_system_instruction_is_non_empty():
    assert SYSTEM_INSTRUCTION
    assert "DriveMind" in SYSTEM_INSTRUCTION