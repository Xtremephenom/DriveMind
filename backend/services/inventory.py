from __future__ import annotations

import shutil
from pathlib import Path

from backend.models.system import (
    DriveInventory,
    FileSystemNode,
    InventoryEntry,
    NodeType,
)


def build_inventory(
    root: FileSystemNode,
    top_n: int = 10,
) -> DriveInventory:
    """
    Analyze a scanned filesystem tree.

    This function is read-only.
    It never modifies or deletes anything.
    """

    if root.node_type != NodeType.DIRECTORY:
        raise ValueError("Inventory root must be a directory.")

    path = Path(root.path)

    usage = shutil.disk_usage(path)

    files = 0
    directories = 0
    junctions = 0
    symlinks = 0
    skipped = 0

    scanned_size = 0

    largest_files: list[InventoryEntry] = []
    largest_directories: list[InventoryEntry] = []

    def walk(node: FileSystemNode, is_root: bool = False) -> None:
        nonlocal files
        nonlocal directories
        nonlocal junctions
        nonlocal symlinks
        nonlocal skipped
        nonlocal scanned_size

        if node.node_type == NodeType.FILE:
            files += 1
            scanned_size += node.size

            largest_files.append(
                InventoryEntry(
                    path=node.path,
                    size=node.size,
                )
            )

        elif node.node_type == NodeType.DIRECTORY:
            if not is_root:
                directories += 1

                largest_directories.append(
                    InventoryEntry(
                        path=node.path,
                        size=node.size,
                    )
                )

        elif node.node_type == NodeType.JUNCTION:
            junctions += 1

        elif node.node_type == NodeType.SYMLINK:
            symlinks += 1

        elif node.node_type == NodeType.SKIPPED:
            skipped += 1

        for child in node.children:
            walk(child)

    walk(root, is_root=True)

    largest_files.sort(
        key=lambda entry: entry.size,
        reverse=True,
    )

    largest_directories.sort(
        key=lambda entry: entry.size,
        reverse=True,
    )

    return DriveInventory(
        path=root.path,

        total_space=usage.total,
        free_space=usage.free,
        used_space=usage.used,

        scanned_size=scanned_size,

        files=files,
        directories=directories,
        junctions=junctions,
        symlinks=symlinks,
        skipped=skipped,

        largest_files=largest_files[:top_n],
        largest_directories=largest_directories[:top_n],
    )


def inventory_to_dict(inventory: DriveInventory) -> dict:
    """
    Convert DriveInventory into a JSON-compatible dictionary.
    """

    return {
        "path": inventory.path,

        "total_space": inventory.total_space,
        "free_space": inventory.free_space,
        "used_space": inventory.used_space,

        "scanned_size": inventory.scanned_size,

        "files": inventory.files,
        "directories": inventory.directories,
        "junctions": inventory.junctions,
        "symlinks": inventory.symlinks,
        "skipped": inventory.skipped,

        "largest_files": [
            {
                "path": entry.path,
                "size": entry.size,
            }
            for entry in inventory.largest_files
        ],

        "largest_directories": [
            {
                "path": entry.path,
                "size": entry.size,
            }
            for entry in inventory.largest_directories
        ],
    }