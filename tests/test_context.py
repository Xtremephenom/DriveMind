from backend.models.system import (
    FileCategory,
    FileEvidence,
    FileRecord,
    Recommendation,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.context import (
    build_decision_context,
    context_to_dict,
)


def test_build_decision_context():
    file = FileRecord(
        path=r"C:\Temp\old.tmp",
        size=5000,
        category=FileCategory.TEMPORARY,
    )

    evidence = FileEvidence(
        path=file.path,
        size=file.size,
        extension=".tmp",
        exists=True,
        age_days=90,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        is_locked=False,
        category=file.category,
        signals=[
            "Temporary file.",
            "Older than 30 days.",
        ],
    )

    recommendation = Recommendation(
        path=file.path,
        size=file.size,
        category=file.category,
        action=RecommendationAction.REVIEW,
        risk=RecommendationRisk.LOW,
        reason="Old temporary file.",
    )

    context = build_decision_context(
        file,
        evidence,
        recommendation,
    )

    assert context.path == file.path
    assert context.size == 5000
    assert context.extension == ".tmp"
    assert context.category == FileCategory.TEMPORARY

    assert context.exists is True
    assert context.age_days == 90

    assert context.is_system_path is False
    assert context.is_locked is False

    assert context.current_action == (
        RecommendationAction.REVIEW
    )

    assert context.current_risk == (
        RecommendationRisk.LOW
    )

    assert len(context.signals) == 2


def test_context_to_dict():
    file = FileRecord(
        path=r"C:\Temp\old.tmp",
        size=5000,
        category=FileCategory.TEMPORARY,
    )

    evidence = FileEvidence(
        path=file.path,
        size=file.size,
        extension=".tmp",
        exists=True,
        age_days=90,
        category=file.category,
        signals=["Old temporary file."],
    )

    recommendation = Recommendation(
        path=file.path,
        size=file.size,
        category=file.category,
        action=RecommendationAction.REVIEW,
        risk=RecommendationRisk.LOW,
        reason="Old temporary file.",
    )

    context = build_decision_context(
        file,
        evidence,
        recommendation,
    )

    data = context_to_dict(context)

    assert data["path"] == file.path
    assert data["size"] == 5000
    assert data["extension"] == ".tmp"
    assert data["category"] == "temporary"

    assert data["exists"] is True
    assert data["age_days"] == 90

    assert data["current_action"] == "review"
    assert data["current_risk"] == "low"

    assert data["signals"] == [
        "Old temporary file."
    ]