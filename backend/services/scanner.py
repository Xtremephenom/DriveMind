from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.models.system import FileSystemNode, NodeType
from backend.utils.filesystem import is_junction, is_symlink


# Generous library defaults: they exist to stop unbounded recursion and
# unbounded allocation, not to constrain a legitimate scan. The dev API
# passes its own, much tighter, configured limits.
DEFAULT_MAX_DEPTH = 64
DEFAULT_MAX_NODES = 1_000_000


class ScanLimitExceeded(RuntimeError):
    """A scan exceeded its depth or node budget."""


@dataclass
class _Budget:
    max_depth: int
    max_nodes: int
    nodes: int = 0

    def consume(self) -> None:
        self.nodes += 1

        if self.nodes > self.max_nodes:
            raise ScanLimitExceeded(
                f"Scan exceeded the maximum of {self.max_nodes} nodes."
            )


def scan_directory(
    root: str | Path,
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> FileSystemNode:
    """
    Recursively scan a directory and build a filesystem tree.

    Files and directories that cannot be accessed are represented
    as skipped nodes rather than crashing the scan.

    A scan that would exceed `max_depth` or `max_nodes` raises
    ScanLimitExceeded rather than silently returning a truncated tree
    whose sizes would understate reality.
    """

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    budget = _Budget(max_depth=max_depth, max_nodes=max_nodes)

    return _scan_directory(root, budget, depth=0)


def _scan_directory(
    directory: Path,
    budget: _Budget,
    depth: int,
) -> FileSystemNode:
    budget.consume()

    result = FileSystemNode(
        path=str(directory),
        node_type=NodeType.DIRECTORY,
        scanned=True,
    )

    if depth >= budget.max_depth:
        raise ScanLimitExceeded(
            f"Scan exceeded the maximum depth of "
            f"{budget.max_depth} at {directory}."
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
                budget.consume()
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
                budget.consume()
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
                budget.consume()

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
                child = _scan_directory(
                    entry,
                    budget,
                    depth + 1,
                )

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