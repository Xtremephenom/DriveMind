from __future__ import annotations

from datetime import datetime
from pathlib import Path

from backend.models.system import (
    FileEvidence,
    FileRecord,
    FileCategory,
)


def build_file_evidence(
    file: FileRecord,
) -> FileEvidence:
    """
    Build factual evidence about a file.

    Path-based evidence is collected even when the file
    no longer exists.

    This function does not modify the filesystem.
    """

    path = Path(file.path)

    extension = path.suffix.lower()

    evidence = FileEvidence(
        path=file.path,
        size=file.size,
        extension=extension,
        category=file.category,
    )

    evidence.exists = path.exists()

    normalized = str(path).lower().replace("/", "\\")

    # ---------------------------------------------------------
    # Path-based evidence
    # ---------------------------------------------------------

    if "\\windows\\" in normalized:
        evidence.is_system_path = True
        evidence.signals.append(
            "Located inside the Windows directory."
        )

    if "\\users\\" in normalized:
        evidence.is_user_path = True
        evidence.signals.append(
            "Located inside a user profile."
        )

    if "\\program files\\" in normalized:
        evidence.is_application_path = True
        evidence.signals.append(
            "Located inside Program Files."
        )

    # ---------------------------------------------------------
    # Classification evidence
    # ---------------------------------------------------------

    if file.category == FileCategory.UNKNOWN:
        evidence.signals.append(
            "No deterministic classification rule matched."
        )

    # ---------------------------------------------------------
    # Filesystem metadata
    # ---------------------------------------------------------

    if evidence.exists:
        try:
            stat = path.stat()

            modified = datetime.fromtimestamp(
                stat.st_mtime
            )

            age = datetime.now() - modified

            evidence.age_days = (
                age.total_seconds() / 86400
            )

        except (PermissionError, OSError):
            evidence.signals.append(
                "File metadata could not be fully accessed."
            )
    else:
        evidence.signals.append(
            "File no longer exists."
        )

    return evidence


def evidence_to_dict(
    evidence: FileEvidence,
) -> dict:
    return {
        "path": evidence.path,
        "size": evidence.size,
        "extension": evidence.extension,
        "age_days": evidence.age_days,
        "exists": evidence.exists,
        "is_locked": evidence.is_locked,
        "is_system_path": evidence.is_system_path,
        "is_user_path": evidence.is_user_path,
        "is_application_path": evidence.is_application_path,
        "category": evidence.category.value,
        "signals": evidence.signals,
    }