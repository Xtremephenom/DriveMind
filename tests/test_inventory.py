from pathlib import Path

from backend.services.inventory import (
    build_inventory,
    inventory_to_dict,
)

from backend.services.scanner import scan_directory


def create_inventory_tree(root: Path) -> None:
    (root / "documents").mkdir()
    (root / "games").mkdir()

    (root / "small.txt").write_text("12345")

    (root / "documents" / "medium.txt").write_text(
        "DriveMind" * 5
    )

    (root / "games" / "large.txt").write_text(
        "X" * 100
    )


def test_build_inventory(tmp_path):
    create_inventory_tree(tmp_path)

    scanned = scan_directory(tmp_path)
    inventory = build_inventory(scanned)

    assert inventory.path == str(tmp_path)

    assert inventory.files == 3
    assert inventory.directories == 2

    assert inventory.junctions == 0
    assert inventory.symlinks == 0
    assert inventory.skipped == 0

    assert inventory.scanned_size == 150


def test_largest_files(tmp_path):
    create_inventory_tree(tmp_path)

    scanned = scan_directory(tmp_path)
    inventory = build_inventory(scanned, top_n=2)

    assert len(inventory.largest_files) == 2

    assert inventory.largest_files[0].path.endswith(
        "large.txt"
    )

    assert inventory.largest_files[0].size == 100

    assert inventory.largest_files[1].path.endswith(
        "medium.txt"
    )


def test_largest_directories(tmp_path):
    create_inventory_tree(tmp_path)

    scanned = scan_directory(tmp_path)
    inventory = build_inventory(scanned)

    assert len(inventory.largest_directories) == 2

    assert inventory.largest_directories[0].path.endswith(
        "games"
    )

    assert inventory.largest_directories[0].size == 100


def test_inventory_to_dict(tmp_path):
    create_inventory_tree(tmp_path)

    scanned = scan_directory(tmp_path)
    inventory = build_inventory(scanned)

    data = inventory_to_dict(inventory)

    assert data["path"] == str(tmp_path)

    assert data["files"] == 3
    assert data["directories"] == 2

    assert isinstance(data["largest_files"], list)
    assert isinstance(data["largest_directories"], list)

    assert "total_space" in data
    assert "free_space" in data
    assert "used_space" in data
    assert "scanned_size" in data