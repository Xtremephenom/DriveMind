"""
DriveMind development HTTP interface.

Development-only (§29/§273). Every request path is untrusted input and
is resolved against an explicit allow-list before the filesystem is
touched at all; with no allow-list configured, every scan is refused.
"""

from fastapi import APIRouter, HTTPException

from backend.core.config import get_settings
from backend.core.scan_scope import ScopeError, resolve_scan_root
from backend.services.analysis import analyze_tree, analysis_to_dict
from backend.services.scanner import (
    ScanLimitExceeded,
    directory_to_dict,
    scan_directory,
)


router = APIRouter()


def _scan_within_scope(path: str):
    """
    Resolve `path` against the allow-list and scan it.

    Translates every failure mode into an HTTPException so both routes
    answer identically — the mapping lived in two copies before.
    """

    settings = get_settings()

    try:
        root = resolve_scan_root(path, settings)
    except ScopeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        return scan_directory(
            root,
            max_depth=settings.max_scan_depth,
            max_nodes=settings.max_scan_nodes,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Directory does not exist.",
        )

    except NotADirectoryError:
        raise HTTPException(
            status_code=400,
            detail="Path is not a directory.",
        )

    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Permission denied.",
        )

    except ScanLimitExceeded as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/scan")
def scan(path: str):
    return directory_to_dict(_scan_within_scope(path))


@router.get("/analyze")
def analyze(path: str):
    return analysis_to_dict(analyze_tree(_scan_within_scope(path)))
