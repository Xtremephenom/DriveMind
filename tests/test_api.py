from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_analyze_endpoint():
    response = client.get(
        "/analyze",
        params={
            "path": r"D:\DriveMind\scanner_test"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_files"] == 3
    assert len(data["files"]) == 3
    assert len(data["recommendations"]) == 3


def test_analyze_missing_directory():
    response = client.get(
        "/analyze",
        params={
            "path": r"D:\DriveMind\does_not_exist"
        },
    )

    assert response.status_code == 404


def test_analyze_file_path():
    response = client.get(
        "/analyze",
        params={
            "path": r"D:\DriveMind\scanner_test\file1.txt"
        },
    )

    assert response.status_code == 400