"""
The read-only guarantee, in a form that can fail.

[`docs/security.md`](../docs/security.md) makes two claims to an
operator's face: that DriveMind does not modify the files it scans, and
that the only filesystem write in the whole package is the dataset build
writing `data/*.jsonl`. Both were originally established by a grep run by
hand, which is a measurement with a shelf life -- correct on the day it
was taken and silently wrong after the next commit.

These tests are the same claims restated as assertions. The first checks
the behaviour against a real tree. The rest check the property at the
source level, which is what catches a write on a code path no behavioural
test happens to exercise.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

from backend.services.analysis import analyze_tree
from backend.services.inventory import build_inventory
from backend.services.scanner import scan_directory


BACKEND = Path(__file__).resolve().parent.parent / "backend"


def _modules() -> list[Path]:
    """
    Every module in the package.

    The count is asserted because a source-level sweep over an empty file
    list passes vacuously, which is the one way these tests could report
    success while checking nothing.
    """

    modules = sorted(BACKEND.rglob("*.py"))

    assert len(modules) > 30, f"only found {len(modules)} modules to sweep"

    return modules


# --- The behaviour: a scan leaves the tree as it found it --------------


def fingerprint(root: Path) -> dict[str, tuple]:
    """
    Modification time, size, and content hash of every entry under `root`.

    Content is hashed rather than inferred from size, because a
    same-length overwrite is precisely the mutation a size comparison
    misses.
    """

    out: dict[str, tuple] = {}

    for path in sorted(root.rglob("*")):
        stat = path.stat(follow_symlinks=False)

        digest = None

        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

        out[str(path.relative_to(root))] = (
            stat.st_mtime_ns,
            stat.st_size,
            digest,
        )

    return out


def test_a_scan_does_not_modify_what_it_reads(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / "empty").mkdir()

    (tmp_path / "file1.txt").write_bytes(b"x" * 100)
    (tmp_path / "sub" / "file2.log").write_bytes(b"y" * 250)
    (tmp_path / "sub" / "deeper" / "file3.dmp").write_bytes(b"z" * 400)

    before = fingerprint(tmp_path)

    tree = scan_directory(tmp_path)
    analysis = analyze_tree(tree)
    inventory = build_inventory(tree)

    # Assert the pass actually read the tree before asserting that it
    # changed nothing. "Nothing changed" is not a guarantee if the scan
    # did nothing.
    assert inventory.scanned_size == 750
    assert inventory.files == 3
    assert analysis.total_files == 3

    after = fingerprint(tmp_path)

    assert set(after) == set(before), "the scan created or removed an entry"
    assert after == before, "the scan altered an mtime, a size, or content"


# --- The property: only the dataset build writes -----------------------


_MUTATORS = frozenset(
    {
        "chmod", "copy", "copy2", "copyfile", "copytree", "hardlink_to",
        "lchmod", "link", "makedirs", "mkdir", "move", "remove",
        "removedirs", "rename", "renames", "rmdir", "rmtree", "symlink",
        "symlink_to", "touch", "truncate", "unlink", "utime",
        "write_bytes", "write_text",
    }
)


# `dataset/writer.write_jsonl` creates `data/` and opens a `.jsonl` for
# writing. It is reached only from the dataset build, and the path it
# writes is handed to it by that build -- never one a scan produced.
_ALLOWED = frozenset(
    {
        ("services/dataset/writer.py", "mkdir"),
        ("services/dataset/writer.py", "open"),
    }
)


def _open_mode(call: ast.Call) -> str:
    """
    The mode an `open()` call was given.

    `open(file, mode)` and `path.open(mode)` put the mode in different
    positions, so which of the two this is has to be settled before the
    argument is read. Getting it wrong fails in the *permissive*
    direction -- an unnoticed `path.open("w")` -- and that is exactly what
    the stale-allow-list assertion below caught the first time this ran.

    A mode assembled at runtime is reported as a writing mode: one we
    cannot read is one we cannot clear.
    """

    index = 0 if isinstance(call.func, ast.Attribute) else 1

    node: ast.expr | None = None

    if len(call.args) > index:
        node = call.args[index]

    for keyword in call.keywords:
        if keyword.arg == "mode":
            node = keyword.value

    if node is None:
        return "r"

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    return "wax"


def _mutating_calls(source: str):
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue

        function = node.func

        if isinstance(function, ast.Attribute):
            name = function.attr
        elif isinstance(function, ast.Name):
            name = function.id
        else:
            continue

        if name in _MUTATORS:
            yield node.lineno, name

        elif name == "open":
            if set(_open_mode(node)) & set("wax+"):
                yield node.lineno, "open"

        elif name == "replace" and len(node.args) < 2:
            # `str.replace(old, new)` is pure. `Path.replace(target)`
            # renames over its destination and takes one argument, so
            # argument count separates them without a type checker.
            yield node.lineno, "replace"


def test_only_the_dataset_build_writes_to_the_filesystem():
    found: set[tuple[str, str]] = set()
    offenders: list[str] = []

    for module in _modules():
        relative = module.relative_to(BACKEND).as_posix()
        source = module.read_text(encoding="utf-8")

        for lineno, name in _mutating_calls(source):
            found.add((relative, name))

            if (relative, name) not in _ALLOWED:
                offenders.append(f"backend/{relative}:{lineno} {name}()")

    assert not offenders, (
        "Unexpected filesystem write in the backend package:\n  "
        + "\n  ".join(offenders)
        + "\nIf it is intentional, docs/security.md and "
        "docs/architecture.md both state that it does not happen and "
        "have to change in the same commit."
    )

    # An allow-list entry that matches nothing guards nothing (§589).
    assert _ALLOWED <= found, (
        f"stale allow-list entries: {sorted(_ALLOWED - found)}"
    )


# --- The property: no shell, no dynamic execution ----------------------


_FORBIDDEN_MODULES = frozenset({"subprocess", "pty", "commands"})

# `compile` is deliberately absent: `re.compile` is everywhere and the
# dynamic-execution risk is already covered by eval / exec / __import__.
_FORBIDDEN_CALLS = frozenset(
    {
        "system", "popen", "eval", "exec", "execv", "execve", "execvp",
        "spawnl", "spawnv", "__import__",
    }
)


def test_the_backend_never_shells_out():
    """
    A scan handles attacker-chosen filenames (§83/§316). A path that hands
    one to a shell or to `eval` is a different class of bug from anything
    the safety gate covers, so it is excluded structurally rather than
    watched for in review.
    """

    offenders: list[str] = []

    for module in _modules():
        relative = module.relative_to(BACKEND).as_posix()

        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                        offenders.append(
                            f"backend/{relative}:{node.lineno} "
                            f"import {alias.name}"
                        )

            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]

                if root in _FORBIDDEN_MODULES:
                    offenders.append(
                        f"backend/{relative}:{node.lineno} from {root}"
                    )

            elif isinstance(node, ast.Call):
                function = node.func

                if isinstance(function, ast.Attribute):
                    name = function.attr
                elif isinstance(function, ast.Name):
                    name = function.id
                else:
                    continue

                if name in _FORBIDDEN_CALLS:
                    offenders.append(
                        f"backend/{relative}:{node.lineno} {name}()"
                    )

    assert not offenders, "\n  ".join(
        ["shell or dynamic-execution surface in the backend:"] + offenders
    )


# --- The property: the HTTP interface is removable ----------------------


# `api/routes.py` and `main.py` are the development HTTP interface
# (§29/§273). Nothing below them may depend on a web framework, or the
# interface stops being a tool and becomes part of the product.
_API_LAYER = frozenset({"api/routes.py", "main.py"})


def test_only_the_api_layer_depends_on_a_third_party_package():
    """
    README.md tells a reader that everything outside the API layer runs on
    the standard library alone. That is what makes the development
    interface deletable rather than load-bearing, so it is worth more than
    a sentence.
    """

    standard_library = set(sys.stdlib_module_names)

    # `__future__` is a real stdlib module, but assert rather than assume
    # it is named here -- a missing entry would make every module look
    # like a third-party dependency and the test would pass for the wrong
    # reason only if the API-layer exemption happened to cover it.
    assert "__future__" in standard_library

    offenders: list[str] = []

    for module in _modules():
        relative = module.relative_to(BACKEND).as_posix()

        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]

            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [(node.module or "").split(".")[0]]

            else:
                continue

            for root in roots:
                if not root or root == "backend":
                    continue

                if root in standard_library:
                    continue

                if relative in _API_LAYER:
                    continue

                offenders.append(
                    f"backend/{relative}:{node.lineno} imports {root}"
                )

    assert not offenders, "\n  ".join(
        ["third-party import outside the API layer:"] + offenders
    )
