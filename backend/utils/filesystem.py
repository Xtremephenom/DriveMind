from __future__ import annotations

from pathlib import Path


def is_symlink(path: Path) -> bool:
    """
    Return True if the path is a symbolic link.
    """
    try:
        return path.is_symlink()
    except (PermissionError, OSError):
        return True


def is_junction(path: Path) -> bool:
    """
    Return True if the path is a Windows NTFS junction.

    Junctions are reparse points that can redirect a directory
    to another location. We do not follow them during scanning.
    """
    try:
        return path.is_junction()
    except (AttributeError, PermissionError, OSError):
        return False