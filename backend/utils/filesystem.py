from __future__ import annotations

from pathlib import Path


# Both probes fail *closed*: when we cannot tell what a path is, we answer
# with whichever value makes the scanner record the node and decline to
# traverse it. Reporting a directory we could not inspect as a redirect we
# did not follow overstates one fact; the alternative -- answering False
# and descending into an unknown reparse point -- risks following a
# redirect out of the scan scope and looping on a cycle. `is_junction`
# answered False on error until this was fixed, so the two probes
# disagreed and the junction path took the unsafe direction (§268: safety
# outranks reporting precision).


def is_symlink(path: Path) -> bool:
    """
    Return True if the path is a symbolic link.

    Returns True when the answer cannot be determined, so an
    undeterminable path is skipped rather than followed.
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

    Returns True when the answer cannot be determined, matching
    `is_symlink`. `AttributeError` is the one exception:
    `Path.is_junction` arrived in Python 3.12, so its absence says the
    interpreter is too old to answer the question at all, not that this
    particular path is suspect.
    """
    try:
        return path.is_junction()
    except AttributeError:
        return False
    except (PermissionError, OSError):
        return True