from __future__ import annotations

from pathlib import Path

from backend.models.system import FileSystemNode, NodeType
from backend.utils.filesystem import is_junction, is_symlink


def scan_directory(root: str | Path) -> FileSystemNode:
    """
    Recursively scan a directory and build a filesystem tree.

    Files and directories that cannot be accessed are represented
    as skipped nodes rather than crashing the scan.
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    return _scan_directory(root)


def _scan_directory(directory: Path) -> FileSystemNode:
    result = FileSystemNode(
        path=str(directory),
        node_type=NodeType.DIRECTORY,
        scanned=True,
    )

    try:
        entries = list(directory.iterdir())
    except (PermissionError, OSError):
        result.scanned = False
        result.reason = "permission_denied"
        return result

    for entry in entries:
        try:
            if is_symlink(entry):
                result.children.append(
                    FileSystemNode(
                        path=str(entry),
                        node_type=NodeType.SYMLINK,
                        scanned=False,
                        reason="symlink_not_followed",
                    )
                )
                continue

            if is_junction(entry):
                result.children.append(
                    FileSystemNode(
                        path=str(entry),
                        node_type=NodeType.JUNCTION,
                        scanned=False,
                        reason="junction_not_followed",
                    )
                )
                continue

            if entry.is_file():
                try:
                    size = entry.stat().st_size

                    result.size += size

                    result.children.append(
                        FileSystemNode(
                            path=str(entry),
                            node_type=NodeType.FILE,
                            size=size,
                            scanned=True,
                        )
                    )

                except (PermissionError, OSError):
                    result.children.append(
                        FileSystemNode(
                            path=str(entry),
                            node_type=NodeType.SKIPPED,
                            scanned=False,
                            reason="file_stat_failed",
                        )
                    )

                continue

            if entry.is_dir():
                child = _scan_directory(entry)

                result.size += child.size
                result.children.append(child)

        except (PermissionError, OSError):
            result.children.append(
                FileSystemNode(
                    path=str(entry),
                    node_type=NodeType.SKIPPED,
                    scanned=False,
                    reason="access_error",
                )
            )

    return result


def directory_to_dict(node: FileSystemNode) -> dict:
    """
    Convert a filesystem node tree into a JSON-compatible dictionary.
    """

    return {
        "path": node.path,
        "type": node.node_type.value,
        "size": node.size,
        "scanned": node.scanned,
        "reason": node.reason,
        "children": [
            directory_to_dict(child)
            for child in node.children
        ],
    }