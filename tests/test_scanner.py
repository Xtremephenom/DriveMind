import os
import subprocess
from pathlib import Path

import pytest

from backend.services.scanner import (
    scan_directory,
    directory_to_dict,
)

from backend.utils.filesystem import (
    is_junction,
    is_symlink,
)

from backend.models.system import NodeType


def create_test_tree(root: Path) -> None:
    (root / "folder1" / "subfolder").mkdir(parents=True)

    (root / "file1.txt").write_text("hello")
    (root / "folder1" / "file2.txt").write_text("DriveMind")
    (root / "folder1" / "subfolder" / "file3.txt").write_text("test")


def test_scan_directory(tmp_path):
    create_test_tree(tmp_path)

    result = scan_directory(tmp_path)

    assert result.node_type == NodeType.DIRECTORY
    assert result.scanned is True
    assert result.size == 18
    assert len(result.children) == 2


def test_scan_directory_returns_correct_path(tmp_path):
    create_test_tree(tmp_path)

    result = scan_directory(tmp_path)

    assert Path(result.path) == tmp_path


def test_directory_to_dict(tmp_path):
    create_test_tree(tmp_path)

    result = scan_directory(tmp_path)
    data = directory_to_dict(result)

    assert data["type"] == "directory"
    assert data["size"] == 18
    assert data["scanned"] is True
    assert data["reason"] is None
    assert len(data["children"]) == 2


def test_missing_directory():
    with pytest.raises(FileNotFoundError):
        scan_directory(r"D:\DriveMind\this_does_not_exist")


def test_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello")

    with pytest.raises(NotADirectoryError):
        scan_directory(file_path)


def test_empty_directory(tmp_path):
    result = scan_directory(tmp_path)

    assert result.node_type == NodeType.DIRECTORY
    
    assert result.size == 0
    assert result.children == []


def test_nested_empty_directories(tmp_path):
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)

    result = scan_directory(tmp_path)

    assert result.size == 0
    assert len(result.children) == 1
    assert result.children[0].node_type == NodeType.DIRECTORY


def test_multiple_files(tmp_path):
    files = {
        "a.txt": "12345",
        "b.txt": "hello",
        "c.txt": "DriveMind",
    }

    for name, content in files.items():
        (tmp_path / name).write_text(content)

    result = scan_directory(tmp_path)

    assert result.size == sum(
        len(content.encode())
        for content in files.values()
    )

    assert all(
        child.node_type == NodeType.FILE
        for child in result.children
    )


def test_normal_directory_is_not_junction(tmp_path):
    assert is_junction(tmp_path) is False


def test_normal_file_is_not_symlink(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("DriveMind")

    assert is_symlink(file_path) is False
def test_junction_is_detected_and_not_followed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()

    (target / "large.txt").write_text("This must not be scanned.")

    junction = tmp_path / "junction"

    result = subprocess.run(
        [
            "cmd",
            "/c",
            "mklink",
            "/J",
            str(junction),
            str(target),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(
            f"Could not create junction: {result.stderr or result.stdout}"
        )

    try:
        assert is_junction(junction) is True

        scanned = scan_directory(tmp_path)

        junction_nodes = [
            child
            for child in scanned.children
            if Path(child.path) == junction
        ]

        assert len(junction_nodes) == 1

        node = junction_nodes[0]

        assert node.node_type == NodeType.JUNCTION
        assert node.scanned is False
        assert node.reason == "junction_not_followed"

        assert node.children == []

    finally:
        subprocess.run(
            [
                "cmd",
                "/c",
                "rmdir",
                str(junction),
            ],
            capture_output=True,
            text=True,
        )
def test_symlink_is_detected_and_not_followed(tmp_path):
    target = tmp_path / "target"
    target.mkdir()

    (target / "file.txt").write_text("Do not follow me.")

    link = tmp_path / "symlink"

    try:
        os.symlink(
            target,
            link,
            target_is_directory=True,
        )
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink creation unavailable: {exc}")

    try:
        assert is_symlink(link) is True

        scanned = scan_directory(tmp_path)

        symlink_nodes = [
            child
            for child in scanned.children
            if Path(child.path) == link
        ]

        assert len(symlink_nodes) == 1

        node = symlink_nodes[0]

        assert node.node_type == NodeType.SYMLINK
        assert node.scanned is False
        assert node.reason == "symlink_not_followed"

        assert node.children == []

    finally:
        try:
            link.unlink()
        except FileNotFoundError:
            pass