"""
Unit tests for scan-scope enforcement.

These cover the guard directly, without a web framework, because the
same guard has to protect the desktop application (§29/§273).
"""

from pathlib import Path

import pytest

from backend.core.config import (
    ENV_ALLOWED_ROOTS,
    DEFAULT_MAX_SCAN_DEPTH,
    DEFAULT_MAX_SCAN_NODES,
    get_settings,
)
from backend.core.scan_scope import ScopeError, resolve_scan_root


REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_TEST = REPO_ROOT / "scanner_test"


def settings_allowing(*roots: Path):
    return get_settings(
        env={ENV_ALLOWED_ROOTS: ";".join(str(r) for r in roots)}
    )


def test_no_allow_list_denies_everything():
    settings = get_settings(env={})

    assert settings.allowed_roots == ()

    with pytest.raises(ScopeError, match="No scan roots"):
        resolve_scan_root(str(SCANNER_TEST), settings)


def test_allowed_root_itself_is_permitted():
    settings = settings_allowing(SCANNER_TEST)

    assert resolve_scan_root(str(SCANNER_TEST), settings) == SCANNER_TEST


def test_descendant_of_allowed_root_is_permitted():
    settings = settings_allowing(SCANNER_TEST)

    target = SCANNER_TEST / "folder1" / "subfolder"

    assert resolve_scan_root(str(target), settings) == target


def test_parent_of_allowed_root_is_refused():
    settings = settings_allowing(SCANNER_TEST)

    with pytest.raises(ScopeError, match="outside every"):
        resolve_scan_root(str(REPO_ROOT), settings)


def test_traversal_escaping_the_root_is_refused():
    settings = settings_allowing(SCANNER_TEST)

    escape = str(SCANNER_TEST / ".." / ".." / "Windows")

    with pytest.raises(ScopeError, match="outside every"):
        resolve_scan_root(escape, settings)


def test_prefix_collision_is_not_treated_as_containment():
    # "scanner_test_other" must not be accepted because it shares a
    # string prefix with "scanner_test".
    settings = settings_allowing(SCANNER_TEST)

    sibling = str(REPO_ROOT / (SCANNER_TEST.name + "_other"))

    with pytest.raises(ScopeError, match="outside every"):
        resolve_scan_root(sibling, settings)


@pytest.mark.parametrize(
    "path, message",
    [
        (r"\\server\share", "UNC"),
        (r"//server/share", "UNC"),
        (r"\\?\C:\Windows", "Device-namespace"),
        (r"\\.\PhysicalDrive0", "Device-namespace"),
        ("", "required"),
        ("   ", "required"),
        ("C:\\Temp\\a\x00b", "null byte"),
    ],
)
def test_malformed_and_special_paths_are_refused(path, message):
    settings = settings_allowing(SCANNER_TEST)

    with pytest.raises(ScopeError, match=message):
        resolve_scan_root(path, settings)


@pytest.mark.parametrize("name", ["NUL", "con", "COM1", "lpt9"])
def test_reserved_device_names_are_refused(name):
    settings = settings_allowing(SCANNER_TEST)

    with pytest.raises(ScopeError, match="reserved device name"):
        resolve_scan_root(str(SCANNER_TEST / name), settings)


def test_forward_slashes_are_accepted_inside_the_root():
    settings = settings_allowing(SCANNER_TEST)

    target = str(SCANNER_TEST).replace("\\", "/") + "/folder1"

    assert resolve_scan_root(target, settings) == (
        SCANNER_TEST / "folder1"
    )


def test_multiple_roots_are_each_honoured():
    other = REPO_ROOT / "data"

    settings = settings_allowing(SCANNER_TEST, other)

    assert resolve_scan_root(str(SCANNER_TEST), settings) == SCANNER_TEST
    assert resolve_scan_root(str(other), settings) == other


def test_limits_fall_back_to_defaults_when_invalid():
    settings = get_settings(
        env={
            ENV_ALLOWED_ROOTS: str(SCANNER_TEST),
            "DRIVEMIND_MAX_SCAN_DEPTH": "0",
            "DRIVEMIND_MAX_SCAN_NODES": "not-a-number",
        }
    )

    assert settings.max_scan_depth == DEFAULT_MAX_SCAN_DEPTH
    assert settings.max_scan_nodes == DEFAULT_MAX_SCAN_NODES


def test_dev_api_host_is_loopback():
    assert get_settings(env={}).dev_api_host == "127.0.0.1"

