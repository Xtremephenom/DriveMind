# ADR 0002 — Deferred deterministic policy hardening

- **Status:** Accepted
- **Date:** 2026-08-27
- **Policy version at time of writing:** `policy-v1` (frozen)
- **The policy itself:** [`docs/policy-v1.md`](../policy-v1.md), enforced by
  `tests/test_policy_v1.py`

## Context

A reconciliation pass against the Master Product Specification found five
gaps in the deterministic decision policy. Fixing any of them changes the
labels the engine emits, which would invalidate the 10,000-example
training dataset, its 8,000/1,000/1,000 split, and the measured
34.82% keep / 65.18% review distribution
([`docs/dataset-card.md`](../dataset-card.md)).

The decision was taken to **freeze `policy-v1`** for this pass: the
dataset stays valid and the gaps are recorded here instead of fixed.

This ADR is the record. It exists so the gaps are not rediscovered from
scratch, and so the safer verdicts encoded in a module that is being
deleted are not lost with it.

## The governing constraint

Every gap below is **latent, not active**, for exactly one reason: the
deterministic engine emits zero `DELETE` recommendations. Its most
permissive verdict is `REVIEW`, which is an instruction to a human, not
an action.

The moment `DELETE` becomes reachable, each of these becomes a
data-loss path. Therefore:

> **Enabling `DELETE` is gated on closing every gap in this ADR.**
> This is a hard precondition on the future cleanup-engine ADR, not a
> recommendation (§443 safety-gate immutability).

## Gap 1 — `is_locked` is dead end to end

`FileEvidence.is_locked` and `DecisionContext.is_locked` both exist.
Neither is ever written by `backend/services/evidence.py:build_file_evidence`,
and neither is ever read by `backend/services/decision/engine.py`.

Consequences measured in-tree:

- A locked, in-use file is indistinguishable from a free one.
- The corpus therefore contains **no** `is_locked: true` rows at all. It
  cannot: the generator derives every field from production code, and
  production never sets the flag. An earlier corpus carried it on 2,691 of
  10,000 rows with **zero** label signal, which taught the model that a
  feature it will never be shown at serving time is irrelevant. Locked
  cases now exist only in `data/red_team.jsonl`, where they are probes
  rather than training signal.
- `backend/services/ai_dataset.py` recorded a `locked-temp` baseline case
  as `KEEP/HIGH`, asserting a locked-file rule the engine does not
  implement. Under a frozen `policy-v1` the engine is authoritative, so
  the *baseline case* was wrong, not the engine. That is corrected by
  deriving baseline labels from the engine rather than hardcoding them.

Closing it requires a share-mode probe (`pywin32`, or `CreateFileW` via
`ctypes`) plus a `KEEP`/`HIGH` locked-file branch, and a dataset
regeneration.

## Gap 2 — `is_system_path` matches only `\windows\`

`backend/services/evidence.py` sets `is_system_path` from the single
substring `\windows\`. It therefore misses:

- `\program files\`
- `\program files (x86)\`
- `\programdata\`

`is_application_path` has the mirror-image bug: it matches
`\program files\` but not `\program files (x86)\`.

Consequence: a file under `C:\Program Files\…` bypasses the system-path
branch in the engine and falls through to the conservative fallback,
`REVIEW/MEDIUM`, instead of the `REVIEW/HIGH` a system path warrants.

## Gap 3 — Four categories have no engine branch

`make_recommendation` has no branch for `CACHE`, `INSTALLER`, `DRIVER`, or
`SYSTEM_DATA`. All four reach the conservative fallback and are reported
as `REVIEW/MEDIUM`.

A second, divergent engine existed at `backend/services/recommender.py`.
It was category-only — it ignored evidence, age, and system paths — was
referenced by nothing except its own test, and used a duplicate
`RiskLevel` enum that compared equal to `RecommendationRisk` only because
both subclass `str`. It has been **deleted** for violating §272 (one
authoritative implementation per behavior).

It nevertheless encoded a *safer* verdict for three of the four missing
categories. That is the part worth keeping, so it is recorded here in
full before the module disappears:

| Category | deleted `recommender.py` | current `decision/engine.py` | Assessment |
|---|---|---|---|
| `SYSTEM_DATA` | `KEEP` / `HIGH` | `REVIEW` / `MEDIUM` (fallback) | **`recommender` safer** — adopt |
| `INSTALLER` | `KEEP` / `HIGH` | `REVIEW` / `MEDIUM` (fallback) | **`recommender` safer** — adopt |
| `DRIVER` | `KEEP` / `HIGH` | `REVIEW` / `MEDIUM` (fallback) | **`recommender` safer** — adopt |
| `CACHE` | `REVIEW` / `MEDIUM` | `REVIEW` / `MEDIUM` (fallback) | Same verdict, but by accident — needs an explicit branch |
| `APPLICATION_DATA` | `REVIEW` / `HIGH` | `KEEP` / `HIGH` | **`engine` safer** — keep engine's |
| `CRASH_DUMP` | `REVIEW` / `LOW` (unconditional) | `REVIEW` / `LOW` if age ≥ 30d, else `MEDIUM` | **`engine` safer** — keep engine's |
| `USER_DATA` | `KEEP` / `HIGH` | `KEEP` / `HIGH` | Agree |
| `TEMPORARY` | `REVIEW` / `MEDIUM` (unconditional) | `REVIEW` / `LOW` if age ≥ 30d, else `MEDIUM` | Engine is more specific |
| `LOG` | `REVIEW` / `MEDIUM` (unconditional) | `REVIEW` / `LOW` if age ≥ 30d, else `MEDIUM` | Engine is more specific |
| `UNKNOWN` | `KEEP` / `HIGH` | `KEEP` / `HIGH` | Agree |

The hardening pass should therefore add explicit `KEEP`/`HIGH` branches
for `SYSTEM_DATA`, `INSTALLER`, and `DRIVER`, and an explicit branch for
`CACHE` — not adopt the deleted module wholesale.

All four categories are now **present** in the training data — an earlier
corpus omitted them entirely, because the generator that built it covered
only six of the ten `FileCategory` values. Presence is not the same as
signal: because all four resolve to the same fallback verdict, they are
present but mutually indistinguishable in the labels. Coverage was the
dataset's problem and is fixed; separating them is this ADR's problem and
is not.

## Gap 4 — Classifier over-breadth

`backend/services/classifier.py`:

- `\appdata\local\microsoft\` → `CACHE`. That subtree is not a cache. It
  contains Outlook `.ost`/`.pst` mail stores and Edge profile data.
  Classifying a mail store as cache is a data-loss path the moment
  `DELETE` exists.
- bare `\logs\` → `LOG`. This matches any directory named `logs`
  anywhere, including a developer's own project output.

Both need anchoring to specific, verified cache and log locations rather
than broad substrings.

## Gap 5 — Rule ordering reports missing system files as reviewable

The system-path rule sits above the not-exists rule, so `exists=False` is
unreachable for anything whose path matches `\windows\`. A file that is not
there is reported as `REVIEW`/`HIGH` — "a human should look at this" — when
the correct answer is that there is nothing to look at.

Found by enumerating the full input space while writing
[`docs/policy-v1.md`](../policy-v1.md); recorded there under "Known
artifacts of the ordering" and asserted in `tests/test_policy_v1.py` so it
stays visible.

Harmless under the governing constraint: `REVIEW` on a phantom file wastes
attention, it does not lose data. The fix is to move the not-exists check
above the category and system-path checks, which changes emitted labels and
therefore belongs to the hardening pass.

## Decision

1. `policy-v1` is frozen. `POLICY_VERSION` is **not** bumped by this pass.
2. The five gaps are recorded here and left unfixed.
3. `recommender.py`, `tests/test_recommender.py`, and the duplicate
   `RiskLevel` enum are deleted; the table above preserves what they knew.
4. `DELETE` remains unreachable, and closing all five gaps is a hard
   precondition on making it reachable.
5. The frozen policy is published as [`docs/policy-v1.md`](../policy-v1.md)
   with a measured truth table, and `tests/test_policy_v1.py` re-measures it
   on every run. An accidental policy change now fails the suite instead of
   silently invalidating the dataset.

## Consequences

- The dataset and split remain valid, and the distribution measured under
  this policy is published in [`docs/dataset-card.md`](../dataset-card.md).
- The fine-tuned model will be trained on a policy with a known
  conservative bias: four categories collapse into `REVIEW/MEDIUM`.
- Agreement metrics measured against `policy-v1` are not comparable to
  metrics measured after the hardening pass. Any recorded baseline must
  state the policy version it was measured against (§514/§516).
- The hardening pass will fail `tests/test_policy_v1.py` by design. That
  failure is the checklist: each expectation it breaks is a published number
  that has to be re-measured.

