# `policy-v1` — the frozen deterministic decision policy

- **Status:** frozen
- **Date recorded:** 2026-08-27
- **Implementation:** [`backend/services/decision/engine.py`](../backend/services/decision/engine.py)
  (`make_recommendation`)
- **Enforced by:** [`tests/test_policy_v1.py`](../tests/test_policy_v1.py)
- **Known gaps:** [ADR 0002](adr/0002-deferred-policy-hardening.md)

This document is the versioned statement of what DriveMind's deterministic
engine decides, and why the version number matters. Every agreement figure,
label distribution, and dataset in the repository is measured *against a
policy version*. Change the policy and those numbers stop being comparable,
which is why this one is frozen and why the next one must be `policy-v2`
with its own document (§514/§516).

The table below was not written from the source. It was **measured** by
enumerating every reachable input combination through `make_recommendation`,
and `tests/test_policy_v1.py` re-measures it on every test run. If the
engine and this document ever disagree, the suite fails.

## What the engine reads

| Field | Read? |
|---|---|
| `FileRecord.category` | yes |
| `FileEvidence.is_system_path` | yes |
| `FileEvidence.exists` | yes |
| `FileEvidence.age_days` | yes |
| `FileEvidence.is_user_path` | **no** |
| `FileEvidence.is_application_path` | **no** |
| `FileEvidence.is_locked` | **no** — see ADR 0002, Gap 1 |
| `FileEvidence.signals` | **no** |
| `FileRecord.size` / `extension` | copied to the output only |

`size` and `path` travel through into the `Recommendation` so the result can
be displayed; they do not influence the verdict. No branch is keyed on file
size, so a 4 KB log and a 40 GB log receive the same recommendation.

## The rule table

Evaluated top to bottom; the first match wins. This ordering is part of the
policy, not an implementation detail — two rules below overlap and the
order is what resolves them.

| # | Condition | Action | Risk |
|---|---|---|---|
| 1 | `category == unknown` | `keep` | `high` |
| 2 | `category == user_data` | `keep` | `high` |
| 3 | `category == application_data` | `keep` | `high` |
| 4 | `is_system_path` | `review` | `high` |
| 5 | `not exists` | `keep` | `high` |
| 6 | `category == temporary` and `age_days >= 30` | `review` | `low` |
| 7 | `category == temporary` | `review` | `medium` |
| 8 | `category == log` and `age_days >= 30` | `review` | `low` |
| 9 | `category == log` | `review` | `medium` |
| 10 | `category == crash_dump` and `age_days >= 30` | `review` | `low` |
| 11 | `category == crash_dump` | `review` | `medium` |
| 12 | anything else | `review` | `medium` |

`age_days is None` (an age that could not be determined) never satisfies the
`>= 30` tests, so it falls to the `medium` variant. Unknown age is treated as
recent, which is the conservative direction.

## The measured truth table

Category × `is_system_path` × `exists` × age. Every cell was produced by
running the engine.

| category | sys | exists | age=None | age=0 | age=29 | age=30 | age=400 |
|---|---|---|---|---|---|---|---|
| `temporary` | no | yes | review/medium | review/medium | review/medium | review/low | review/low |
| `temporary` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `temporary` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `cache` | no | yes | review/medium | review/medium | review/medium | review/medium | review/medium |
| `cache` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `cache` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `log` | no | yes | review/medium | review/medium | review/medium | review/low | review/low |
| `log` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `log` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `crash_dump` | no | yes | review/medium | review/medium | review/medium | review/low | review/low |
| `crash_dump` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `crash_dump` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `installer` | no | yes | review/medium | review/medium | review/medium | review/medium | review/medium |
| `installer` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `installer` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `driver` | no | yes | review/medium | review/medium | review/medium | review/medium | review/medium |
| `driver` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `driver` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `system_data` | no | yes | review/medium | review/medium | review/medium | review/medium | review/medium |
| `system_data` | no | no | keep/high | keep/high | keep/high | keep/high | keep/high |
| `system_data` | yes | — | review/high | review/high | review/high | review/high | review/high |
| `user_data` | — | — | keep/high | keep/high | keep/high | keep/high | keep/high |
| `application_data` | — | — | keep/high | keep/high | keep/high | keep/high | keep/high |
| `unknown` | — | — | keep/high | keep/high | keep/high | keep/high | keep/high |

## Properties this policy guarantees

1. **`delete` is unreachable.** No branch produces
   `RecommendationAction.DELETE`. DriveMind cannot currently recommend
   deleting anything, and therefore cannot mis-recommend it. Every gap in
   ADR 0002 is latent *because of this property*, and enabling `delete` is
   gated on closing all of them.
2. **Action and risk are independent.** `review/high` is a normal, frequent
   verdict: "this needs a human, and getting it wrong would hurt." It is not
   a contradiction and must never be collapsed into a single score (§23.7).
3. **`user_data`, `application_data`, and `unknown` are unconditional.**
   Rules 1–3 sit above every other rule, so no combination of age, system
   path, or non-existence can move them off `keep/high`. This is deliberate:
   the categories where a wrong answer is unrecoverable are decided by
   category alone.
4. **The default is `review/medium`, never `delete` or `keep`.** A file the
   policy has no rule for is surfaced to the user, not silently dismissed
   and not silently actioned.

## Known artifacts of the ordering

- **A non-existent file inside a system path returns `review/high`, not
  `keep/high`.** Rule 4 precedes rule 5, so `exists=False` is never reached
  for system paths. Recommending "review" for a file that is not there is
  meaningless but harmless under property 1. Recorded rather than fixed,
  because fixing it changes emitted labels.
- **Four categories have no rule of their own.** `cache`, `installer`,
  `driver`, and `system_data` all fall to rule 12. They are indistinguishable
  from each other and from an unclassifiable-but-categorised file. This is
  ADR 0002 Gap 3. All four are present in the training corpus, but because
  the engine that labels the data has nothing specific to say about them,
  they are present without being separable — a model cannot learn a
  distinction the labels do not contain.
- **`is_locked` has no effect.** A locked, in-use temporary file receives
  exactly the verdict an idle one does. ADR 0002 Gap 1. Because the flag
  carries no label signal and production never sets it, the training corpus
  omits it entirely; it appears only in the red-team set.

## Changing this policy

Any change to `make_recommendation` — including one that looks like a pure
improvement — requires all of:

1. a new `docs/policy-v2.md` with its own measured truth table,
2. regeneration of `data/train|validation|test.jsonl`, because the labels
   come from the engine (§65),
3. re-measurement of every published distribution and agreement figure, with
   the policy version stated alongside each one, and
4. an ADR recording what changed and which prior measurements it voided.

Model metrics measured against `policy-v1` are not comparable to metrics
measured against any later policy. There is no shortcut around this.
