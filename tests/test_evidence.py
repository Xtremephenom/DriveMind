from pathlib import Path

from backend.models.system import (
    FileCategory,
    FileRecord,
)

from backend.services.evidence import (
    build_file_evidence,
    evidence_to_dict,
)


def make_file(
    path: str,
    size: int = 1000,
    category: FileCategory = FileCategory.UNKNOWN,
) -> FileRecord:
    return FileRecord(
        path=path,
        size=size,
        category=category,
    )


def test_evidence_for_existing_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("DriveMind")

    file = make_file(
        str(file_path),
        size=file_path.stat().st_size,
    )

    evidence = build_file_evidence(file)

    assert evidence.exists is True
    assert evidence.size == 9
    assert evidence.extension == ".txt"
    assert evidence.age_days is not None


def test_system_path_signal():
    file = make_file(
        r"C:\Windows\Temp\test.tmp",
        category=FileCategory.TEMPORARY,
    )

    evidence = build_file_evidence(file)

    assert evidence.is_system_path is True
    assert "Located inside the Windows directory." in evidence.signals


def test_user_path_signal():
    file = make_file(
        r"C:\Users\Test\Documents\test.pdf",
        category=FileCategory.USER_DATA,
    )

    evidence = build_file_evidence(file)

    assert evidence.is_user_path is True
    assert "Located inside a user profile." in evidence.signals


def test_program_files_signal():
    file = make_file(
        r"C:\Program Files\TestApp\data.bin",
        category=FileCategory.APPLICATION_DATA,
    )

    evidence = build_file_evidence(file)

    assert evidence.is_application_path is True


def test_unknown_category_signal():
    file = make_file(
        r"C:\somewhere\unknown.xyz"
    )

    evidence = build_file_evidence(file)

    assert (
        "No deterministic classification rule matched."
        in evidence.signals
    )


def test_missing_file():
    file = make_file(
        r"C:\definitely\does\not\exist\file.xyz"
    )

    evidence = build_file_evidence(file)

    assert evidence.exists is False
    assert evidence.age_days is None
    assert "File no longer exists." in evidence.signals


def test_evidence_to_dict():
    file = make_file(
        r"C:\Windows\Temp\test.tmp",
        size=5000,
        category=FileCategory.TEMPORARY,
    )

    evidence = build_file_evidence(file)
    data = evidence_to_dict(evidence)

    assert data["path"] == file.path
    assert data["size"] == 5000
    assert data["category"] == "temporary"
    assert isinstance(data["signals"], list)