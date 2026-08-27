"""
Configuration for DriveMind's development interface.

The FastAPI surface is a development tool, not the product (§29/§273),
so everything it is allowed to touch is declared here rather than being
implied by whatever path a caller happens to send.

The defaults are deliberately hostile: with no configuration at all the
allow-list is empty and every scan request is refused. Granting access
is an explicit act.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# The dev API must never be reachable from another machine.
DEV_API_HOST = "127.0.0.1"
DEV_API_PORT = 8000

ENV_ALLOWED_ROOTS = "DRIVEMIND_ALLOWED_ROOTS"
ENV_MAX_SCAN_DEPTH = "DRIVEMIND_MAX_SCAN_DEPTH"
ENV_MAX_SCAN_NODES = "DRIVEMIND_MAX_SCAN_NODES"

DEFAULT_MAX_SCAN_DEPTH = 24
DEFAULT_MAX_SCAN_NODES = 200_000


@dataclass(frozen=True)
class Settings:
    """Resolved development-interface settings."""

    allowed_roots: tuple[Path, ...]
    max_scan_depth: int
    max_scan_nodes: int
    dev_api_host: str = DEV_API_HOST
    dev_api_port: int = DEV_API_PORT


def _parse_allowed_roots(raw: str | None) -> tuple[Path, ...]:
    if not raw or not raw.strip():
        return ()

    roots: list[Path] = []

    for entry in raw.split(os.pathsep):
        entry = entry.strip().strip('"')

        if not entry:
            continue

        try:
            resolved = Path(entry).resolve(strict=False)
        except (OSError, ValueError):
            # An unresolvable configured root grants nothing.
            continue

        if resolved not in roots:
            roots.append(resolved)

    return tuple(roots)


def _parse_positive_int(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw.strip())
    except ValueError:
        return default

    if value < 1:
        return default

    return value


def get_settings(env: dict[str, str] | None = None) -> Settings:
    """
    Read settings from the environment.

    Read on every call rather than cached, so a test or a desktop host
    can change the allow-list without reaching into module state.
    """

    source = os.environ if env is None else env

    return Settings(
        allowed_roots=_parse_allowed_roots(
            source.get(ENV_ALLOWED_ROOTS)
        ),
        max_scan_depth=_parse_positive_int(
            source.get(ENV_MAX_SCAN_DEPTH),
            DEFAULT_MAX_SCAN_DEPTH,
        ),
        max_scan_nodes=_parse_positive_int(
            source.get(ENV_MAX_SCAN_NODES),
            DEFAULT_MAX_SCAN_NODES,
        ),
    )
