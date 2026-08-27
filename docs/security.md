# Security

Operator-facing companion to [`docs/threat-model.md`](threat-model.md).
The threat model analyses what could go wrong; this document states what
the code actually guarantees, what you must configure, and what is
explicitly not protected.

## The one thing to know

**The HTTP interface in `backend/api/` and `backend/main.py` is a
development tool, not a product surface (§29/§273).** It has no
authentication. Do not expose it, do not put it behind a reverse proxy to
"make it available", and do not build a shipping feature on it. It exists
so a developer can exercise the backend by hand.

## Guarantees

These are properties of the code, each with a test that fails if it stops
holding.

### DriveMind does not modify scanned files

Asserted at two levels by
[`tests/test_read_only.py`](../tests/test_read_only.py), not by review:

- **Behaviourally.** A tree is fingerprinted by mtime, size, **and content
  hash**, then scanned, analysed, and inventoried, then fingerprinted
  again. Content is hashed rather than inferred from size, because a
  same-length overwrite is exactly what a size comparison misses. The test
  also asserts the pass actually read the tree first — "nothing changed" is
  not a guarantee if the scan did nothing.
- **At the source level**, which is what catches a write on a code path no
  behavioural test happens to run. Every module in the package is parsed
  and swept for 25 mutating calls (`unlink`, `rmtree`, `rename`,
  `write_text`, `chmod`, `utime`, `copytree`, …), for `open()` in any
  writing mode, and for single-argument `.replace()` — `Path.replace`
  renames over its destination, while the two-argument `str.replace` is
  pure. There is exactly **one** permitted write, named explicitly:
  `dataset/writer.write_jsonl`, which creates `data/*.jsonl` inside the
  repository at a path the dataset build hands it. The allow-list is itself
  asserted non-stale, so an entry that stops matching fails the test rather
  than silently widening it.

Two further structural properties, same file: the backend imports no
`subprocess`/`pty` and calls no `eval`/`exec`/`os.system`/`os.popen` — a
scan handles attacker-chosen filenames, and handing one to a shell is a
class of bug the safety gate does not cover — and no module outside
`api/routes.py` and `main.py` imports a third-party package, which is what
makes the development HTTP interface deletable rather than load-bearing.

There is no cleanup engine: `policy-v1` cannot emit a `DELETE`
recommendation at all, proven across the full input space by
`tests/test_policy_v1.py::test_delete_is_unreachable`. Every recommendation
DriveMind produces today is a statement, not an action.

The scan itself reads metadata (`stat`, `iterdir`) and never file contents.

### DriveMind does not phone home

No telemetry, no analytics, no crash reporting, no update check, no model
download. There is no HTTP client in `backend/` — the only network-capable
dependency is the dev server, which listens on loopback and initiates
nothing. Nothing about a user's disk leaves the machine by any code path in
this repository.

The training corpus in `data/` is entirely synthetic. No real filesystem
was scanned to produce it (see [`docs/dataset-card.md`](dataset-card.md)).

### A path outside the allow-list is refused before the filesystem is touched

`core/scan_scope.resolve_scan_root` is the single gate, and it has no
bypass — both routes go through `_scan_within_scope`. It refuses, in order:
an unconfigured allow-list, a blank path, a null byte, a `\\?\` or `\\.\`
device-namespace path, a `\\` UNC path, a reserved device name (`CON`,
`NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`) in any component, an unresolvable
path, and anything that resolves outside every configured root.

Two details that make it hold rather than merely appear to:

- **`resolve()` runs before the containment check**, so `..` is collapsed
  *and symlinks are followed*. A link inside an allowed root that points at
  `C:\Users` resolves outside the root and is rejected. Checking the string
  first is how this control usually fails.
- **Containment is `resolved == root or root in resolved.parents`**, not a
  string prefix. `C:\AllowedEvil` does not pass with root `C:\Allowed`.

`tests/test_scan_scope.py` covers each refusal.

### The safety gate cannot be widened by its input

`ai_safety.validate_ai_response` calls `recommend_for_context` and computes
its own ceiling from the case's evidence. It does not read
`case.context.current_action`. A case whose `DecisionContext` has been
tampered to `current_action=DELETE` still clamps to the engine's real
verdict — asserted in `tests/test_ai_safety.py`. The reasoning is
[ADR 0001](adr/0001-deterministic-authority.md).

### A path we cannot classify is not traversed

Both reparse-point probes in `utils/filesystem.py` fail **closed**: a
`PermissionError` or `OSError` from `is_junction()` or `is_symlink()`
answers `True`, so the node is recorded and its contents are never read.

### Traversal is bounded

`ScanLimitExceeded` is raised — never silently truncated — when a scan
exceeds its depth or node budget. Truncation would produce a smaller number
that looks like a complete measurement.

| | Library default | Dev API default |
|---|---|---|
| depth | 64 | **24** |
| nodes | 1,000,000 | **200,000** |

## What you must configure

### `DRIVEMIND_ALLOWED_ROOTS` — required, and a security control

**Empty or unset means every scan is refused**, including of the repository
itself. This is not a bug to work around; the insecure default would be
"allow everything", and this default is the opposite.

Separate multiple roots with `os.pathsep` (`;` on Windows). Entries are
resolved at read time; an unresolvable entry grants nothing rather than
failing open.

```bash
DRIVEMIND_ALLOWED_ROOTS='D:\DriveMind\scanner_test' .venv/Scripts/python.exe -m backend.main
```

Grant the narrowest root that does the job. Setting it to `C:\` authorizes
exactly the filename disclosure that T1 in the threat model describes — a
legitimate configuration for a disk analyser, and entirely your decision,
but make it knowingly.

### `DRIVEMIND_MAX_SCAN_DEPTH` / `DRIVEMIND_MAX_SCAN_NODES` — optional

Default 24 and 200,000. A non-numeric or non-positive value falls back to
the default rather than disabling the cap.

### Host and port

Not configurable by environment, deliberately. `DEV_API_HOST` is
`127.0.0.1` in `core/config.py`. If you find yourself editing that
constant, the thing you want is not this interface.

## What is *not* protected

Stated here rather than left for someone to discover:

- **No authentication, no authorization, no CSRF defence.** Any local
  process — and any web page the user visits — can `GET
  /scan?path=<allowed root>` while the server runs. Accepted only because
  the interface is development-only. A shipping product must not use it.
- **No transport encryption.** Plain HTTP on loopback.
- **No rate limiting, no request size limits, no timeouts.** A single
  request can consume a CPU core for the duration of a large scan.
- **No streaming.** The full tree is built in memory, then serialized. The
  node cap bounds the allocation; it does not make it incremental.
- **The explanation text is untrusted.** When a model provider is wired up,
  `Recommendation.reason` can be model-authored prose derived from
  filenames an attacker chose. Any UI must render it as inert text — no
  HTML, no markdown links, no auto-linking — and must keep the action and
  risk visually dominant over it. See T3 in the threat model.
- **No dependency hash pinning, lockfile, or SBOM.** Four exact version
  pins, no `--require-hashes`, no audit step.
- **No long-path (`\\?\`) handling.** Paths beyond `MAX_PATH` are not
  specially handled, and `\\?\` input is *rejected* by the scope gate. A
  deeply nested real directory may fail to scan (§128).

## Reporting honesty

Treated as a security property here because a false claim about
reclaimable space causes the same harm as a wrong recommendation.

`AnalysisResult` exposes `review_size` and `deletable_size` separately and
provides no combined total. `deletable_size` is always 0 today, which is the
truthful answer rather than an omission. The removed
`total_recommended_size` summed `REVIEW` bytes and presented them as
reclaimable; `tests/test_analysis.py` asserts the name is absent from both
the object and the serialized output so it cannot return by accident.

## Secrets

None are required. No API keys, no tokens, no credentials — configuration
is three non-secret environment variables, and nothing in `backend/` writes
a log file.

When the ML track begins, Kaggle and Hugging Face credentials must live in
the environment or in Kaggle's secret store — never in the repository,
never in a committed notebook cell, never in printed output (§287/§427).

## Reporting a vulnerability

This is a pre-release repository with no published release, no users, and no
security contact. Open an issue, or fix it and say what you fixed.
