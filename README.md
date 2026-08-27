# DriveMind

Local-first Windows disk intelligence. It answers *what is taking up space
on this machine, and what is safe to remove?* — and it is architected so
that the second half of that question is answered by deterministic code, not
by a language model.

**Status: pre-release.** There is no installer, no GUI, and no cleanup
engine. Nothing in this repository deletes a file; `DELETE` is currently
unreachable by construction. What works today is the scan → classify →
evidence → recommend pipeline, the AI safety architecture around it, and a
10,000-row synthetic training corpus. See
[What works today](#what-works-today).

## Five principles

Every design decision in here traces to one of these, and where they
conflict the earlier one wins.

1. **Understanding** — explain what a file *is* before saying anything about
   removing it.
2. **Evidence** — every recommendation cites the observations behind it.
3. **Safety** — a wrong `KEEP` costs disk space; a wrong `DELETE` can cost
   something irreplaceable. The asymmetry decides every close call.
4. **Privacy** — nothing about the user's disk leaves the machine. No
   telemetry, no analytics, no network client.
5. **User control** — DriveMind recommends. The user decides.

## Prerequisites

- **Windows.** Junction detection, path semantics, and the classifier's
  directory rules are Windows-specific.
- **Python 3.12 or newer.** Development runs on **3.14.2**; 3.12.0 is the
  lowest version the code has been byte-compiled and imported under, so
  that is what `pyproject.toml` declares. Nothing lower has been tested,
  and claiming otherwise would be a guess presented as a requirement.

> **If you are on the author's machine:** the default `python` on `PATH` is
> `D:\PYTHON 3.12\python.exe` and it **has no pytest installed**. Every
> command below invokes the venv interpreter *by path* for that reason.
> `python -m pytest` will fail with `No module named pytest`, which looks
> like a broken repository and is not.

## Setup

```bash
git clone <this repository> && cd DriveMind
```

```bash
py -3.14 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

3.14 is what development uses and what the test count below was measured
on. `py -3.12` also works — that is the declared floor — but nothing has
been measured there beyond confirming every module byte-compiles and
imports.

Four pinned dependencies, nothing else: `fastapi`, `uvicorn` (the
development HTTP interface), `pytest`, and `httpx2` (required by
`fastapi.testclient`). Everything in `backend/` outside the API layer runs
on the standard library alone.

## Run the tests

This is the check that tells you the clone is good:

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected on a fresh clone: **438 passed, 4 skipped**. Nothing has to be
created by hand first — the suite builds every tree it scans in a temporary
directory, so a clone is good or bad on its own.

Three of the four skips are the dataset contract tests, which have nothing
to read until `data/` exists (`.gitignore` excludes it). Build it — see
[Build the training corpus](#build-the-training-corpus) — and they run,
giving **441 passed, 1 skipped**.

The remaining skip is `test_symlink_is_detected_and_not_followed`, which
needs `SeCreateSymbolicLinkPrivilege` — enable Developer Mode or use an
elevated shell to run it. It skips cleanly rather than failing. The junction
equivalent does run, because `mklink /J` needs no privilege.

Test count is not a vanity metric here. Most of what this codebase asserts
is a safety property — that a model cannot escalate an action, that an
undeterminable path is not traversed, that a prompt does not contain the
answer — and a safety property with no test is a comment. If one of those
fails, the fix is the code, never the expectation.

## Run a scan

Read-only. Reads metadata only, never file contents.

`scanner_test/` is a local fixture, not a committed one — `.gitignore`
excludes it, so a fresh clone has to create it:

```bash
mkdir -p scanner_test/folder1/subfolder && printf 'file one' > scanner_test/file1.txt && printf 'file two xyz' > scanner_test/folder1/file2.txt && printf 'file  3' > scanner_test/folder1/subfolder/file3.txt
```

```bash
.venv/Scripts/python.exe -c "import json; from backend.services.scanner import scan_directory; from backend.services.analysis import analyze_tree, analysis_to_dict; print(json.dumps(analysis_to_dict(analyze_tree(scan_directory('scanner_test'))), indent=2))"
```

That tree reports 3 files, all
`category: unknown` → `action: keep, risk: high`, with
`review_size: 0` and `deletable_size: 0`.

`unknown → keep` is the correct answer, not a gap: no classification rule
matched, so DriveMind declines to recommend removal and says why.

Those two size fields are never summed and there is no combined total.
`review_size` is bytes a human needs to look at; `deletable_size` is bytes
that can actually be freed, and it is `0` today because the policy emits no
`DELETE`. Reporting the first as the second conflates action with risk, and
an earlier version of this code did exactly that.

## Run the AI baseline

```bash
.venv/Scripts/python.exe -m backend.services.ai_baseline
```

Reports structured-output validity, action agreement, risk agreement, and
unsafe escalation rate over 12 cases, and prints its own caveat first: the
only provider that exists is `RuleBasedAIProvider`, a deterministic mirror
of the engine. Its 100% / 100% / 100% / 0% measures the harness, not any AI
capability. Details and the bar a real model has to clear:
[`docs/model-card.md`](docs/model-card.md).

## Build the training corpus

```bash
.venv/Scripts/python.exe -c "import json; from backend.services.dataset.build import build_dataset; print(json.dumps(build_dataset(), indent=2))"
```

Writes `data/{train,validation,test,gold,red_team}.jsonl` — 8,000 / 1,000 /
1,000 / 100 / 26 rows. Deterministic from seed 42, so a rebuild is
byte-identical unless the policy or generator changed.

Every label comes from running the real engine on generated evidence;
nothing hardcodes an action or a risk. The build **refuses to write** on
duplicate `case_id`, cross-file overlap, a missing `FileCategory`, a signal
outside the production vocabulary, or a prompt containing the answer — a bad
corpus is a worse outcome than no corpus.
[`docs/dataset-card.md`](docs/dataset-card.md) has the measured
distributions and six honest limitations.

## Start the development HTTP interface

**This is a development tool, not a product surface.** It has no
authentication. Do not expose it.

```bash
DRIVEMIND_ALLOWED_ROOTS='D:\DriveMind\scanner_test' .venv/Scripts/python.exe -m backend.main
```

Binds `127.0.0.1:8000`. Then `GET /scan?path=...` or `GET /analyze?path=...`.

**`DRIVEMIND_ALLOWED_ROOTS` is required.** Unset or empty, *every* scan is
refused — the default denies rather than allows. Separate multiple roots
with `;`. Optional: `DRIVEMIND_MAX_SCAN_DEPTH` (24),
`DRIVEMIND_MAX_SCAN_NODES` (200,000).

Verified against the real scope gate with the root above:

| Request | Result |
|---|---|
| `scanner_test` | 200 |
| `scanner_test\..\..` | 400 outside every configured scan root |
| `C:\Windows` | 400 outside every configured scan root |
| `D:\DriveMind\scanner_test_evil` | 400 — containment is not a string prefix |
| `\\server\share` | 400 network (UNC) paths are not accepted |
| `\\?\C:\Windows`, `\\.\PhysicalDrive0` | 400 device-namespace paths are not accepted |
| `scanner_test\NUL` | 400 reserved device name |
| path with a null byte | 400 null byte |
| blank path | 400 a path is required |

## Repository map

```
backend/
  api/routes.py          GET /scan, GET /analyze          (development only)
  main.py                dev server, loopback bind
  core/config.py         settings; defaults deny
  core/scan_scope.py     the untrusted-path gate
  models/system.py       every dataclass and enum
  services/
    scanner.py           read-only traversal, depth + node budget
    inventory.py         tree -> totals, counts, largest-N
    classifier.py        path/extension -> one of ten categories
    evidence.py          path shape, existence, age -> evidence + signals
    decision/engine.py   THE AUTHORITY: make_recommendation, POLICY_VERSION
    analysis.py          scan tree -> recommendations
    context.py           evidence + verdict, travelling together
    ai_prompt.py         the leak-free prompt serializer
    ai_response.py       strict parser: {action, risk, explanation}
    ai_safety/validator.py  the gate; re-derives its own ceiling
    ai_review.py         the only path a model may touch a recommendation
    ai_evaluator.py      four metrics against re-derived ground truth
    ai_baseline.py       runnable baseline harness (12 hand-built cases)
    ai_holdout.py        the held-out measurement; verifies the split first
    ai_rule_based.py     deterministic stand-in; NOT a model
    dataset/             generator, labeler, writer, split, scenarios, build
  utils/filesystem.py    reparse-point probes; both fail closed
data/                    synthetic corpus (5 files, 10,126 rows)
docs/                    architecture, security, threat model, cards, ADRs
ml/                      model runtime, held-out eval CLIs, QLoRA fine-tune
tests/                   27 modules, 4,175 lines
```

39 modules, 4,207 lines under `backend/`. The tests are slightly larger
than the code they cover.

`ml/` is outside `backend/` on purpose: it is the only part of the
repository that imports `torch` and `transformers`, and
`tests/test_read_only.py::test_only_the_api_layer_depends_on_a_third_party_package`
fails if anything under `backend/` ever does. Its four modules, 2,244
lines, are not needed to run the product or its tests, and
`ml/requirements.txt` is deliberately separate from `requirements.txt`.

## The architectural claim

The one thing worth understanding before changing anything:

```
FILESYSTEM -> SCANNER -> CLASSIFIER -> EVIDENCE
  -> DETERMINISTIC POLICY     <-- authoritative
  -> AI CASE -> PROMPT        <-- the verdict does not cross here
  -> LOCAL LLM                <-- advisory, untrusted
  -> STRICT PARSER
  -> SAFETY GATE              <-- re-derives its own ceiling, clamps
  -> FINAL RECOMMENDATION
```

The deterministic engine produces a complete recommendation before a model
is consulted at all. The model sees evidence, never the verdict. Its answer
may only make the result **more** conservative: it can turn `REVIEW` into
`KEEP` and `LOW` risk into `HIGH`, never the reverse. The gate computes its
own limit by re-running the engine rather than reading it from the case, so
a tampered case cannot widen it. Any provider failure — exception, prose
instead of JSON, an invalid enum — lands the user on the deterministic
answer.

The model's contribution is language, under supervision. That is the
ceiling by design. Full reasoning and the rejected alternatives:
[ADR 0001](docs/adr/0001-deterministic-authority.md).

## What works today

| | |
|---|---|
| Read-only scan with depth/node budgets, junction + symlink containment | ✅ |
| Ten-category classifier, six-signal evidence builder | ✅ |
| Deterministic twelve-rule policy, versioned as `policy-v1` | ✅ |
| AI subordination chain end-to-end, gate, deterministic fallback | ✅ |
| Confined development HTTP interface | ✅ |
| 10,126-row synthetic corpus with build-time integrity gates | ✅ |
| Baseline evaluation harness, four metrics | ✅ |
| **A fine-tuned model** | ❌ nothing trained |
| **A cleanup / delete engine** | ❌ `DELETE` unreachable by construction |
| **A GUI** | ❌ |
| **Long-path (`\\?\`) handling, scan cancellation, progress reporting** | ❌ |
| **`is_locked` — the field exists, nothing writes it** | ❌ ADR 0002 Gap 1 |

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | pipeline stages, layers, what calls what |
| [`docs/policy-v1.md`](docs/policy-v1.md) | the twelve rules, in order, and how to change them |
| [`docs/security.md`](docs/security.md) | guarantees, required configuration, what is *not* protected |
| [`docs/threat-model.md`](docs/threat-model.md) | nine threats, mitigations, residual risk |
| [`docs/dataset-card.md`](docs/dataset-card.md) | corpus provenance, distributions, six limitations |
| [`docs/model-card.md`](docs/model-card.md) | there is no model; what one must report before it counts |
| [`docs/adr/0001-deterministic-authority.md`](docs/adr/0001-deterministic-authority.md) | why the engine decides and the model advises |
| [`docs/adr/0002-deferred-policy-hardening.md`](docs/adr/0002-deferred-policy-hardening.md) | five known policy gaps, all hard blockers on enabling `DELETE` |

## Before you change the policy

`backend/services/decision/engine.py` is the authority, and its verdicts are
stamped onto all 10,126 corpus rows as `policy-v1`. Changing any verdict
means: bump `POLICY_VERSION`, publish `docs/policy-v<n>.md`, regenerate the
corpus, and re-measure every number in the dataset card. Metrics from
different policy versions are not comparable and must not be shown as a
trend. The procedure is in [`docs/policy-v1.md`](docs/policy-v1.md).

The safety gate is not a tuning knob. It must not be weakened to improve
model accuracy, increase deletion volume, make a benchmark look better, or
simplify an implementation.
