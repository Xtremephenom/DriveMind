# Architecture

DriveMind answers one question — *what is taking up space, and what is safe
to remove?* — with a hard rule about who is allowed to answer it. This
document maps the code onto that rule.

Two orthogonal decompositions, both real in the tree:

- **The pipeline** — DISCOVER → UNDERSTAND → RECOMMEND → EXECUTE. What
  happens to a file, in order.
- **The layers** — PRESENTATION → APPLICATION SERVICES → DOMAIN LOGIC →
  INFRASTRUCTURE. What may call what.

Measured, not estimated: 38 modules under `backend/`, 3,839 lines; 27 test
modules, 4,147 lines. Tests are somewhat larger than the code they cover,
which is deliberate — most of what this codebase asserts is a
safety property, and a safety property with no test is a comment.

## The pipeline

### DISCOVER — what is on disk

| Module | Responsibility |
|---|---|
| [`backend/services/scanner.py`](../backend/services/scanner.py) | Recursive traversal into a `FileSystemNode` tree, with a depth and node budget |
| [`backend/utils/filesystem.py`](../backend/utils/filesystem.py) | The two reparse-point probes, `is_symlink` and `is_junction` |
| [`backend/services/inventory.py`](../backend/services/inventory.py) | Aggregates a scanned tree into totals, counts, and largest-N lists |

The scanner is **read-only**, and so is every module in the DISCOVER →
UNDERSTAND → RECOMMEND path. That is asserted, not reviewed:
[`tests/test_read_only.py`](../tests/test_read_only.py) fingerprints a tree
by mtime, size, and content hash across a full scan → analyse → inventory
pass, and separately parses every module in the package looking for 25
mutating calls, `open()` in a writing mode, and single-argument
`.replace()`. There is exactly **one** permitted write —
`dataset/writer.write_jsonl`, which creates `data/*.jsonl` inside the
repository at a path the dataset build supplies. Nothing writes to, moves,
or deletes a path that a scan produced. The precise wording matters more
than a blanket "read-only" would: the guarantee is about scanned paths, and
it is checkable.

The scanner records junctions and symlinks as nodes with `scanned=False`
and does not follow them, so a redirect cannot take a scan outside its root
or spin on a cycle.

It raises `ScanLimitExceeded` rather than truncating when it hits
`max_depth` (default 64) or `max_nodes` (default 1,000,000). Truncation
would produce a smaller number that looks like a complete measurement,
which is the failure mode §555 is about; a raised error cannot be
misread.

`build_inventory` calls `shutil.disk_usage`, so `total_space` / `free_space`
are the volume's own figures while `scanned_size` is what DriveMind
actually walked. They are reported as separate fields and never
reconciled into one, because the difference between them *is* the
information — it is everything the scan could not see.

### UNDERSTAND — what the files are

| Module | Responsibility |
|---|---|
| [`backend/services/classifier.py`](../backend/services/classifier.py) | Path and extension → one of ten `FileCategory` values |
| [`backend/services/evidence.py`](../backend/services/evidence.py) | Path shape, existence, age, and category → `FileEvidence` + human-readable signals |

`_classify_path` is an ordered rule chain, and the order is load-bearing:
temp directories → crash-dump directories → `.dmp`/`.hdmp` → cache → log →
installer → driver → application → system → `\users\` → `UNKNOWN`. Earlier
rules win. `C:\Windows\Temp\x.log` is `TEMPORARY`, not `LOG` or
`SYSTEM_DATA`, because the temp rule fires first. The consequences of that
ordering are enumerated in [`docs/policy-v1.md`](policy-v1.md) under "Known
artifacts of the ordering" rather than left for a reader to rediscover.

`UNKNOWN` is a real answer, not a failure. It means no rule matched, it
carries the signal `No deterministic classification rule matched.`, and
`policy-v1` sends it to `REVIEW`. Guessing would be worse.

Evidence carries six signal strings, and only those six —
`SIGNAL_VOCABULARY` in `evidence.py` is the closed set. The dataset build
fails if a training row uses a signal production cannot emit, or omits one
it can, which is what keeps the training distribution honest about the
serving distribution (§78).

### RECOMMEND — what to do about them

| Module | Responsibility |
|---|---|
| [`backend/services/decision/engine.py`](../backend/services/decision/engine.py) | `make_recommendation` — the authoritative twelve-rule policy, and `POLICY_VERSION` |
| [`backend/services/context.py`](../backend/services/context.py) | `DecisionContext`: evidence + the current verdict, travelling together |
| [`backend/services/ai_cases.py`](../backend/services/ai_cases.py) | `build_case_id` — content-addresses a context by sha256 of its evidence |
| [`backend/services/ai_prompt.py`](../backend/services/ai_prompt.py) | The leak-free prompt serializer |
| [`backend/services/ai.py`](../backend/services/ai.py) | The `AIProvider` interface — one method, `analyze` |
| [`backend/services/ai_response.py`](../backend/services/ai_response.py) | Strict parser: `{action, risk, explanation}`, nothing more, nothing less |
| [`backend/services/ai_safety/validator.py`](../backend/services/ai_safety/validator.py) | The gate |
| [`backend/services/ai_review.py`](../backend/services/ai_review.py) | The only path on which a model may touch a recommendation |
| [`backend/services/analysis.py`](../backend/services/analysis.py) | Scan tree → per-file recommendations → `AnalysisResult` |

This is where the product's central architectural decision lives, and it
has its own document: [ADR 0001](adr/0001-deterministic-authority.md). In
one line — **the engine decides, the model may only argue for something
more conservative, and the gate re-derives its own ceiling rather than
reading it from the case.**

Two details that are easy to lose in a refactor and are each held by a
test:

- **`context_to_file_and_evidence` deliberately drops `current_action` and
  `current_risk`.** That is how the gate re-derives instead of trusting.
  Preserving them "for convenience" would silently turn the ceiling back
  into an input.
- **`build_ai_prompt` is the only serializer that may reach a model.**
  `context_to_dict` includes the verdict — correctly, it is the
  ground-truth serializer — and must never be used to build a prompt.

`AnalysisResult` exposes `review_size` and `deletable_size` as two separate
properties and no combined total. Summing them, or naming either
"recommended", would report bytes awaiting a human decision as bytes the
user can free — the action/risk conflation §23.7 forbids.

### EXECUTE — deleting things

**Does not exist.** No module in `backend/` deletes, moves, quarantines, or
modifies a user file. `policy-v1` cannot even emit a `DELETE`
recommendation (`tests/test_policy_v1.py::test_delete_is_unreachable`
proves it across the whole input space).

This is a stage of the product that is not built yet, and the five
preconditions on building it are recorded in
[ADR 0002](adr/0002-deferred-policy-hardening.md). Every recommendation
DriveMind currently produces is a statement, not an action.

## The layers

Dependencies point downward only.

```
PRESENTATION          api/routes.py, main.py            development only (§29/§273)
        |
APPLICATION SERVICES  analysis.py, inventory.py, ai_review.py,
                      ai_baseline.py, dataset/*
        |
DOMAIN LOGIC          classifier.py, evidence.py, decision/engine.py,
                      context.py, ai_cases.py, ai_prompt.py,
                      ai_response.py, ai_safety/validator.py
        |
INFRASTRUCTURE        scanner.py, utils/filesystem.py,
                      core/config.py, core/scan_scope.py
```

Domain logic is pure: given the same `FileRecord` and `FileEvidence`,
`make_recommendation` returns the same `Recommendation`, with no clock, no
filesystem, and no network. That is what makes 10,000 labels reproducible
from a seed, and what makes `POLICY_VERSION` a meaningful thing to stamp
on a row.

`backend/models/system.py` sits outside the stack — every layer imports its
dataclasses and enums, and it imports nothing from `backend`.

The PRESENTATION boundary is enforced rather than documented:
`tests/test_read_only.py::test_only_the_api_layer_depends_on_a_third_party_package`
walks every module's imports and fails if anything outside `api/routes.py`
and `main.py` reaches for a non-stdlib package. That is what keeps the
development HTTP interface deletable instead of load-bearing — the three
layers below it run on the standard library alone.

The one asymmetry worth knowing about: the classifier treats
`\program files\` and `\programdata\` as system directories, but
`evidence.build_path_evidence` sets `is_system_path` only for `\windows\`.
So `C:\Program Files\…` is classified as system data while *not* carrying
the system-path flag the policy gates on. That is ADR 0002 Gap 2, and it is
latent only because no `DELETE` is reachable.

## The AI subordination chain

Fixed, single-path, in `ai_review.review_file`:

```
FILESYSTEM
  -> SCANNER              scan_directory
  -> CLASSIFIER           classify_file
  -> EVIDENCE             build_file_evidence
  -> DETERMINISTIC POLICY make_recommendation      <-- authoritative
  -> AI CASE              build_decision_context + build_ai_case
  -> PROMPT               build_ai_prompt          <-- no verdict crosses here
  -> LOCAL LLM            AIProvider.analyze       <-- advisory, untrusted
  -> RESPONSE PARSING     parse_ai_response        <-- strict; raises on drift
  -> SAFETY GATE          validate_ai_response     <-- re-derives its ceiling
  -> FINAL RECOMMENDATION
```

Any failure from `analyze` onward — exception, prose instead of JSON, an
invalid enum, an unexpected key, an answer about a different case — lands
the user on the deterministic recommendation from step 4 (§544). There is
no path where a provider failure becomes a user-visible error.

The only provider that exists today is
[`ai_rule_based.py`](../backend/services/ai_rule_based.py), a deterministic
mirror of the engine used to exercise the harness. Its 100% agreement
figures measure the plumbing, not a model — see
[`docs/model-card.md`](model-card.md).

## Supporting subsystems

**Dataset** — [`backend/services/dataset/`](../backend/services/dataset/):
`generator.py` enumerates the evidence space (13,020 combinations),
`labeler.py` labels each one by calling the real engine, `writer.py`
renders rows, `split.py` partitions deterministically, `scenarios.py` holds
the 26 curated red-team probes, and `build.py` writes the five files behind
a set of integrity assertions that refuse the write rather than emit a bad
corpus. Documented in [`docs/dataset-card.md`](dataset-card.md).

**Evaluation** — `ai_evaluator.py` scores one response against ground truth
(structured-output validity, action agreement, risk agreement, unsafe
escalation); `ai_baseline.py` is the runnable harness over
`ai_dataset.build_baseline_cases()`. Unsafe escalation rate is the primary
safety metric, and it is counted *before* the gate clamps — a blocked
escalation is still an escalation the model attempted, and hiding it behind
the gate would make the gate's own effectiveness unmeasurable.

**Development HTTP interface** — `api/routes.py` and `main.py`, confined by
`core/config.py` and `core/scan_scope.py`. It is not a shipping surface;
[`docs/security.md`](security.md) states exactly what confines it and
[`docs/threat-model.md`](threat-model.md) states what it would expose
unconfined.

## What is not here yet

Stated plainly so the map is not mistaken for the territory:

- No GUI. The four-layer separation exists so a Windows client can sit on
  the service layer, but no such client is written.
- No cleanup engine, no quarantine, no undo journal. See EXECUTE above.
- No fine-tuned model. The ML pipeline is specified and the corpus is
  built; nothing has been trained.
- No long-path (`\\?\`) handling, no scan cancellation, no progress
  reporting, single-threaded traversal.
- `is_locked` exists on `FileEvidence`, is written by nothing, and is read
  by nothing (ADR 0002 Gap 1).
