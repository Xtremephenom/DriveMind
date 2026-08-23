from backend.models.system import (
    FileCategory,
    FileRecord,
    RecommendationAction,
    RiskLevel,
)

from backend.services.recommender import (
    recommend,
    recommendation_to_dict,
)


def make_file(
    category: FileCategory,
    size: int = 1000,
) -> FileRecord:
    return FileRecord(
        path=r"C:\test\example.dat",
        size=size,
        category=category,
    )


def test_unknown_file_is_kept():
    result = recommend(
        make_file(FileCategory.UNKNOWN)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RiskLevel.HIGH


def test_user_data_is_kept():
    result = recommend(
        make_file(FileCategory.USER_DATA)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RiskLevel.HIGH


def test_driver_is_kept():
    result = recommend(
        make_file(FileCategory.DRIVER)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RiskLevel.HIGH


def test_installer_is_kept():
    result = recommend(
        make_file(FileCategory.INSTALLER)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RiskLevel.HIGH


def test_system_data_is_kept():
    result = recommend(
        make_file(FileCategory.SYSTEM_DATA)
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RiskLevel.HIGH


def test_temp_requires_review():
    result = recommend(
        make_file(FileCategory.TEMPORARY)
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RiskLevel.MEDIUM


def test_cache_requires_review():
    result = recommend(
        make_file(FileCategory.CACHE)
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RiskLevel.MEDIUM


def test_crash_dump_requires_review():
    result = recommend(
        make_file(FileCategory.CRASH_DUMP)
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RiskLevel.LOW


def test_log_requires_review():
    result = recommend(
        make_file(FileCategory.LOG)
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RiskLevel.MEDIUM


def test_application_data_requires_review():
    result = recommend(
        make_file(FileCategory.APPLICATION_DATA)
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RiskLevel.HIGH


def test_recommendation_to_dict():
    result = recommend(
        make_file(
            FileCategory.CRASH_DUMP,
            size=5000,
        )
    )

    data = recommendation_to_dict(result)

    assert data["size"] == 5000
    assert data["category"] == "crash_dump"
    assert data["action"] == "review"
    assert data["risk"] == "low"
    assert isinstance(data["reason"], str)