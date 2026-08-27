"""
Scan-scope enforcement (§113).

A caller-supplied path is untrusted input. This module turns one into a
concrete directory the scanner is permitted to read, or refuses it.

Deliberately free of any web-framework import so the same guard serves
the desktop application (§29/§273).
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.core.config import Settings


class ScopeError(ValueError):
    """A requested path is outside the permitted scan scope."""


# CON, NUL, COM1 … addressed as files are devices, not directories.
_RESERVED_NAMES = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$",
    re.IGNORECASE,
)


def _reject(message: str) -> ScopeError:
    return ScopeError(message)


def resolve_scan_root(
    requested: str,
    settings: Settings,
) -> Path:
    """
    Resolve `requested` to a directory inside the configured allow-list.

    Raises ScopeError for anything else — including the case where no
    allow-list is configured, which denies everything.
    """

    if not settings.allowed_roots:
        raise _reject(
            "No scan roots are configured. Set "
            "DRIVEMIND_ALLOWED_ROOTS to grant access."
        )

    if requested is None or not requested.strip():
        raise _reject("A path is required.")

    if "\x00" in requested:
        raise _reject("Path contains a null byte.")

    normalized = requested.strip().replace("/", "\\")

    if normalized.startswith("\\\\?\\") or normalized.startswith(
        "\\\\.\\"
    ):
        raise _reject("Device-namespace paths are not accepted.")

    if normalized.startswith("\\\\"):
        raise _reject("Network (UNC) paths are not accepted.")

    candidate = Path(normalized)

    for part in candidate.parts:
        if _RESERVED_NAMES.match(part):
            raise _reject(
                f"Path contains a reserved device name: {part!r}."
            )

    try:
        # resolve() also collapses `..` and follows symlinks, so a link
        # inside an allowed root that points outside it is rejected too.
        resolved = candidate.resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise _reject("Path could not be resolved.") from exc

    for root in settings.allowed_roots:
        if resolved == root or root in resolved.parents:
            return resolved

    raise _reject(
        "Path is outside every configured scan root."
    )
