from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import ENV_ALLOWED_ROOTS
from backend.main import app


client = TestClient(app)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_TEST = REPO_ROOT / "scanner_test"


@pytest.fixture(autouse=True)
def allow_scanner_test(monkeypatch):
    """Grant the dev API access to the fixture tree, and only to it."""

    monkeypatch.setenv(ENV_ALLOWED_ROOTS, str(SCANNER_TEST))


def test_analyze_endpoint():
    response = client.get(
        "/analyze",
        params={"path": str(SCANNER_TEST)},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_files"] == 3
    assert len(data["files"]) == 3
    assert len(data["recommendations"]) == 3


def test_analyze_missing_directory():
    response = client.get(
        "/analyze",
        params={"path": str(SCANNER_TEST / "does_not_exist")},
    )

    assert response.status_code == 404


def test_analyze_file_path():
    response = client.get(
        "/analyze",
        params={"path": str(SCANNER_TEST / "file1.txt")},
    )

    assert response.status_code == 400


# --- Confinement (§113/§285) -------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    ["/scan", "/analyze"],
)
@pytest.mark.parametrize(
    "path",
    [
        r"C:\Windows",
        r"..\..",
        r"\\server\share",
        r"\\?\C:\Windows",
        r"\\.\PhysicalDrive0",
        "",
        "   ",
    ],
)
def test_paths_outside_scope_are_refused(endpoint, path):
    response = client.get(endpoint, params={"path": path})

    assert response.status_code == 400
    assert "children" not in response.text
    assert "recommendations" not in response.text


@pytest.mark.parametrize("endpoint", ["/scan", "/analyze"])
def test_traversal_out_of_an_allowed_root_is_refused(endpoint):
    escape = str(SCANNER_TEST / ".." / ".." / "Windows")

    response = client.get(endpoint, params={"path": escape})

    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/scan", "/analyze"])
def test_empty_allow_list_denies_everything(endpoint, monkeypatch):
    monkeypatch.delenv(ENV_ALLOWED_ROOTS, raising=False)

    response = client.get(endpoint, params={"path": str(SCANNER_TEST)})

    assert response.status_code == 400
    assert "DRIVEMIND_ALLOWED_ROOTS" in response.json()["detail"]


def test_depth_cap_is_enforced(monkeypatch):
    monkeypatch.setenv("DRIVEMIND_MAX_SCAN_DEPTH", "1")

    response = client.get("/scan", params={"path": str(SCANNER_TEST)})

    assert response.status_code == 400
    assert "depth" in response.json()["detail"]


def test_node_cap_is_enforced(monkeypatch):
    monkeypatch.setenv("DRIVEMIND_MAX_SCAN_NODES", "1")

    response = client.get("/scan", params={"path": str(SCANNER_TEST)})

    assert response.status_code == 400
    assert "nodes" in response.json()["detail"]


def test_scan_inside_scope_still_works():
    response = client.get("/scan", params={"path": str(SCANNER_TEST)})

    assert response.status_code == 200
    assert response.json()["scanned"] is True

