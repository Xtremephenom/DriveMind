from backend.models.system import (
    FileCategory,
    FileEvidence,
    FileRecord,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.decision.engine import (
    make_recommendation,
)


def make_file(
    path: str,
    category: FileCategory,
    size: int = 1000,
) -> FileRecord:
    return FileRecord(
        path=path,
        size=size,
        category=category,
    )


def make_evidence(
    file: FileRecord,
    system_path: bool = False,
) -> FileEvidence:
    return FileEvidence(
        path=file.path,
        size=file.size,
        extension=".tmp",
        is_system_path=system_path,
        category=file.category,
    )


def test_unknown_is_keep():
    file = make_file(
        r"C:\unknown.xyz",
        FileCategory.UNKNOWN,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "keep"
    assert result.risk == "high"


def test_user_data_is_keep():
    file = make_file(
        r"C:\Users\Test\Documents\data.pdf",
        FileCategory.USER_DATA,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "keep"
    assert result.risk == "high"


def test_application_data_is_keep():
    file = make_file(
        r"C:\Program Files\TestApp\data.bin",
        FileCategory.APPLICATION_DATA,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "keep"
    assert result.risk == "high"


def test_system_path_requires_review():
    file = make_file(
        r"C:\Windows\Temp\test.tmp",
        FileCategory.TEMPORARY,
    )

    result = make_recommendation(
        file,
        make_evidence(file, system_path=True),
    )

    assert result.action == "review"
    assert result.risk == "high"


def test_temporary_file_requires_review():
    file = make_file(
        r"C:\Temp\test.tmp",
        FileCategory.TEMPORARY,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "review"
    assert result.risk == "medium"


def test_log_requires_review():
    file = make_file(
        r"C:\Logs\application.log",
        FileCategory.LOG,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "review"
    assert result.risk == "medium"


def test_crash_dump_requires_review():
    file = make_file(
        r"C:\Dumps\crash.dmp",
        FileCategory.CRASH_DUMP,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "review"
    assert result.risk == "medium"


def test_unknown_category_defaults_to_review():
    file = make_file(
        r"C:\something.dat",
        FileCategory.CACHE,
    )

    result = make_recommendation(
        file,
        make_evidence(file),
    )

    assert result.action == "review"
    assert result.risk == "medium"
def test_old_temporary_file_is_low_risk_review():
    file = make_file(
        r"C:\Temp\old.tmp",
        FileCategory.TEMPORARY,
    )

    evidence = make_evidence(file)
    evidence.age_days = 60

    result = make_recommendation(
        file,
        evidence,
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW


def test_recent_temporary_file_is_medium_risk():
    file = make_file(
        r"C:\Temp\recent.tmp",
        FileCategory.TEMPORARY,
    )

    evidence = make_evidence(file)
    evidence.age_days = 2

    result = make_recommendation(
        file,
        evidence,
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.MEDIUM


def test_old_log_is_low_risk_review():
    file = make_file(
        r"C:\Logs\old.log",
        FileCategory.LOG,
    )

    evidence = make_evidence(file)
    evidence.age_days = 90

    result = make_recommendation(
        file,
        evidence,
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW


def test_old_crash_dump_is_low_risk_review():
    file = make_file(
        r"C:\Dumps\old.dmp",
        FileCategory.CRASH_DUMP,
    )

    evidence = make_evidence(file)
    evidence.age_days = 120

    result = make_recommendation(
        file,
        evidence,
    )

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW


def test_missing_file_is_keep():
    file = make_file(
        r"C:\Temp\gone.tmp",
        FileCategory.TEMPORARY,
    )

    evidence = make_evidence(file)
    evidence.exists = False

    result = make_recommendation(
        file,
        evidence,
    )

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH