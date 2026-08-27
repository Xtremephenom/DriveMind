from __future__ import annotations

from datetime import datetime
from pathlib import Path

from backend.models.system import (
    FileEvidence,
    FileRecord,
    FileCategory,
)


# The complete signal vocabulary. Signals are shown to the model verbatim,
# so the set of strings that can appear is part of the serving contract:
# the training corpus must use exactly these and no others, or the model
# is being taught a vocabulary it will not be spoken to in (§78).
#
# Named constants rather than inline literals so the wording has one home
# and `tests/test_dataset_build.py` can assert coverage against it.
SIGNAL_IN_WINDOWS_DIRECTORY = "Located inside the Windows directory."
SIGNAL_IN_USER_PROFILE = "Located inside a user profile."
SIGNAL_IN_PROGRAM_FILES = "Located inside Program Files."
SIGNAL_UNCLASSIFIED = "No deterministic classification rule matched."
SIGNAL_MISSING = "File no longer exists."
SIGNAL_METADATA_UNREADABLE = (
    "File metadata could not be fully accessed."
)

SIGNAL_VOCABULARY = frozenset(
    {
        SIGNAL_IN_WINDOWS_DIRECTORY,
        SIGNAL_IN_USER_PROFILE,
        SIGNAL_IN_PROGRAM_FILES,
        SIGNAL_UNCLASSIFIED,
        SIGNAL_MISSING,
        SIGNAL_METADATA_UNREADABLE,
    }
)


def build_path_evidence(
    file: FileRecord,
) -> FileEvidence:
    """
    Build the evidence that can be read from a path alone.

    No filesystem access. `exists` is left at its default and `age_days`
    at `None`; a caller that can observe the file fills those in with
    `apply_existence_evidence`.

    This is split out from `build_file_evidence` so the synthetic dataset
    generator can derive the same flags and the same signal wording that
    production derives. When the generator invented its own, the model was
    trained on evidence it would never be shown at serving time (§78).

    A file can legitimately end up with no signals at all: an ordinary
    temporary file at `C:\\Temp\\x.tmp` with a readable age matches none of
    the conditions below. An empty `signals` list is therefore not a bug,
    which is why the dataset build checks the signal *vocabulary* rather
    than requiring every row to carry one.
    """

    path = Path(file.path)

    evidence = FileEvidence(
        path=file.path,
        size=file.size,
        extension=path.suffix.lower(),
        category=file.category,
    )

    normalized = str(path).lower().replace("/", "\\")

    # ---------------------------------------------------------
    # Path-based evidence
    # ---------------------------------------------------------

    if "\\windows\\" in normalized:
        evidence.is_system_path = True
        evidence.signals.append(SIGNAL_IN_WINDOWS_DIRECTORY)

    if "\\users\\" in normalized:
        evidence.is_user_path = True
        evidence.signals.append(SIGNAL_IN_USER_PROFILE)

    if "\\program files\\" in normalized:
        evidence.is_application_path = True
        evidence.signals.append(SIGNAL_IN_PROGRAM_FILES)

    # ---------------------------------------------------------
    # Classification evidence
    # ---------------------------------------------------------

    if file.category == FileCategory.UNKNOWN:
        evidence.signals.append(SIGNAL_UNCLASSIFIED)

    return evidence


def apply_existence_evidence(
    evidence: FileEvidence,
    *,
    exists: bool,
    age_days: float | None,
) -> FileEvidence:
    """
    Record what was observed about the file itself.

    Mutates and returns `evidence`. Kept separate from
    `build_path_evidence` so an observation can come either from a real
    `stat` call or from a synthetic scenario, with identical results.

    The signals are derived from the observed state rather than from how
    the observation was made. "Exists but age unknown" means the same
    thing whether a `stat` call raised or a synthetic case says so, and
    deriving it here is what makes the generated evidence identical to
    the evidence production emits (§78).
    """

    evidence.exists = exists
    evidence.age_days = age_days

    if not exists:
        evidence.signals.append(SIGNAL_MISSING)

    elif age_days is None:
        evidence.signals.append(SIGNAL_METADATA_UNREADABLE)

    return evidence


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

    evidence = build_path_evidence(file)

    exists = path.exists()
    age_days = None

    # ---------------------------------------------------------
    # Filesystem metadata
    # ---------------------------------------------------------

    if exists:
        try:
            stat = path.stat()

            modified = datetime.fromtimestamp(
                stat.st_mtime
            )

            age = datetime.now() - modified

            age_days = age.total_seconds() / 86400

        except (PermissionError, OSError):
            # Left as None. `apply_existence_evidence` turns that into
            # the "could not be fully accessed" signal, so the wording
            # has one home.
            age_days = None

    return apply_existence_evidence(
        evidence,
        exists=exists,
        age_days=age_days,
    )


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
