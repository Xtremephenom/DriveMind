from __future__ import annotations

from pathlib import Path

from backend.models.system import (
    FileCategory,
    FileRecord,
    FileSystemNode,
    NodeType,
)


TEMP_DIRECTORIES = {
    "\\windows\\temp\\",
    "\\appdata\\local\\temp\\",
}

CACHE_DIRECTORIES = {
    "\\appdata\\local\\microsoft\\",
    "\\appdata\\local\\google\\chrome\\user data\\default\\cache\\",
    "\\appdata\\local\\google\\chrome\\user data\\default\\code cache\\",
}

LOG_DIRECTORIES = {
    "\\windows\\logs\\",
    "\\logs\\",
}

CRASH_DIRECTORIES = {
    "\\windows\\livekernelreports\\",
    "\\windows\\minidump\\",
}

INSTALLER_DIRECTORIES = {
    "\\windows\\installer\\",
}

DRIVER_DIRECTORIES = {
    "\\esupport\\edriver\\",
}

SYSTEM_DIRECTORIES = {
    "\\windows\\",
    "\\program files\\",
    "\\program files (x86)\\",
    "\\programdata\\",
}

APPLICATION_DIRECTORIES = {
    "\\appdata\\local\\",
    "\\appdata\\roaming\\",
}


def classify_file(node: FileSystemNode) -> FileRecord:
    """
    Classify a single filesystem file using deterministic rules.

    This function never deletes, moves, or modifies anything.
    """

    if node.node_type != NodeType.FILE:
        raise ValueError("classify_file() requires a file node.")

    path = Path(node.path)

    normalized = str(path).lower()
    extension = path.suffix.lower() or None

    category, reason = _classify_path(
        normalized,
        extension,
    )

    return FileRecord(
        path=node.path,
        size=node.size,
        category=category,
        extension=extension,
        reason=reason,
    )


def _classify_path(
    normalized_path: str,
    extension: str | None,
) -> tuple[FileCategory, str]:

    for directory in TEMP_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.TEMPORARY,
                "File is located in a known Windows temporary directory.",
            )

    for directory in CRASH_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.CRASH_DUMP,
                "File is located in a Windows crash-dump directory.",
            )

    if extension in {".dmp", ".hdmp"}:
        return (
            FileCategory.CRASH_DUMP,
            "File has a Windows crash-dump extension.",
        )

    for directory in CACHE_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.CACHE,
                "File is located in a known application cache directory.",
            )

    for directory in LOG_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.LOG,
                "File is located in a known log directory.",
            )

    for directory in INSTALLER_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.INSTALLER,
                "File is located in the Windows Installer cache.",
            )

    for directory in DRIVER_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.DRIVER,
                "File is located in an OEM driver package directory.",
            )

    for directory in APPLICATION_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.APPLICATION_DATA,
                "File is located inside a user's application-data directory.",
            )

    for directory in SYSTEM_DIRECTORIES:
        if directory in normalized_path:
            return (
                FileCategory.SYSTEM_DATA,
                "File is located in a Windows/system software directory.",
            )

    if "\\users\\" in normalized_path:
        return (
            FileCategory.USER_DATA,
            "File is located inside a user profile.",
        )

    return (
        FileCategory.UNKNOWN,
        "No deterministic classification rule matched this file.",
    )