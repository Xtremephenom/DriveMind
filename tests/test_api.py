import pytest
from fastapi.testclient import TestClient

from backend.core.config import ENV_ALLOWED_ROOTS
from backend.main import app


client = TestClient(app)


@pytest.fixture(scope="session")
def scanner_tree(tmp_path_factory):
    """
    The three-file tree these tests scan, built here rather than read
    from `scanner_test/` in the repository.

    That directory is gitignored, so a fresh clone does not have it, and
    five of the tests below failed with 404 until someone ran the `mkdir`
    buried in the README's "Run a scan" section. A suite that passes only
    on a machine which has already been used by hand is not the check
    that tells you a clone is good.

    The route resolves scope from the path alone but then stats the
    target, which is why these five need a real directory while
    `test_scan_scope.py` does not -- the gate is path arithmetic, the 404
    is filesystem truth.

    Three details the assertions below depend on: three files in total,
    `file1.txt` directly at the root for the path-is-a-file case, and two
    levels of nesting so the depth cap has something to refuse.
    """

    root = tmp_path_factory.mktemp("scanner_tree")

    (root / "folder1" / "subfolder").mkdir(parents=True)

    (root / "file1.txt").write_text("file one", encoding="utf-8")

    (root / "folder1" / "file2.txt").write_text(
        "file two xyz",
        encoding="utf-8",
    )

    (root / "folder1" / "subfolder" / "file3.txt").write_text(
        "file  3",
        encoding="utf-8",
    )

    return root


@pytest.fixture(autouse=True)
def allow_scanner_tree(monkeypatch, scanner_tree):
    """Grant the dev API access to the fixture tree, and only to it."""

    monkeypatch.setenv(ENV_ALLOWED_ROOTS, str(scanner_tree))


def test_analyze_endpoint(scanner_tree):
    response = client.get(
        "/analyze",
        params={"path": str(scanner_tree)},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_files"] == 3
    assert len(data["files"]) == 3
    assert len(data["recommendations"]) == 3


def test_analyze_missing_directory(scanner_tree):
    response = client.get(
        "/analyze",
        params={"path": str(scanner_tree / "does_not_exist")},
    )

    assert response.status_code == 404


def test_analyze_file_path(scanner_tree):
    response = client.get(
        "/analyze",
        params={"path": str(scanner_tree / "file1.txt")},
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
def test_traversal_out_of_an_allowed_root_is_refused(endpoint, scanner_tree):
    escape = str(scanner_tree / ".." / ".." / "Windows")

    response = client.get(endpoint, params={"path": escape})

    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/scan", "/analyze"])
def test_empty_allow_list_denies_everything(
    endpoint,
    monkeypatch,
    scanner_tree,
):
    monkeypatch.delenv(ENV_ALLOWED_ROOTS, raising=False)

    response = client.get(endpoint, params={"path": str(scanner_tree)})

    assert response.status_code == 400
    assert "DRIVEMIND_ALLOWED_ROOTS" in response.json()["detail"]


def test_depth_cap_is_enforced(monkeypatch, scanner_tree):
    monkeypatch.setenv("DRIVEMIND_MAX_SCAN_DEPTH", "1")

    response = client.get("/scan", params={"path": str(scanner_tree)})

    assert response.status_code == 400
    assert "depth" in response.json()["detail"]


def test_node_cap_is_enforced(monkeypatch, scanner_tree):
    monkeypatch.setenv("DRIVEMIND_MAX_SCAN_NODES", "1")

    response = client.get("/scan", params={"path": str(scanner_tree)})

    assert response.status_code == 400
    assert "nodes" in response.json()["detail"]


def test_scan_inside_scope_still_works(scanner_tree):
    response = client.get("/scan", params={"path": str(scanner_tree)})

    assert response.status_code == 200
    assert response.json()["scanned"] is True
