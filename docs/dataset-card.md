# Dataset card — DriveMind synthetic decision corpus

- **Built:** 2026-08-27
- **Policy version:** `policy-v1` (frozen) — [`docs/policy-v1.md`](policy-v1.md)
- **Generator:** [`backend/services/dataset/generator.py`](../backend/services/dataset/generator.py)
- **Red-team seed:** [`backend/services/dataset/scenarios.py`](../backend/services/dataset/scenarios.py)
- **Build + integrity gate:** [`backend/services/dataset/build.py`](../backend/services/dataset/build.py)
- **Seed:** `42`
- **Repository state at build time:** `5c27bc3` plus the uncommitted Phase 0–3
  working tree. Reproducible from the seed by re-running the command below;
  the commit hash must be re-recorded here once this pass is committed.

Rebuild:

```bash
.venv/Scripts/python.exe -c "import json; from backend.services.dataset.build import build_dataset; print(json.dumps(build_dataset(), indent=2))"
```

## What a row is

```json
{
  "case_id": "<16 hex chars, sha256 of the evidence>",
  "policy_version": "policy-v1",
  "prompt": "<system instruction + CASE json + closing instruction>",
  "response": {"action": "keep|review", "risk": "low|medium|high",
               "explanation": "<one sentence>"}
}
```

`case_id` is metadata for tooling. It is **not** in the prompt and the model
is never asked to echo it, so the prompt is exactly the evidence. That is
what makes prompt-level leakage detection meaningful: identical evidence
renders as an identical string instead of being disguised by a different
hash.

`policy_version` is `backend.services.decision.engine.POLICY_VERSION`,
stamped on every row rather than recorded once in this file, because rows
travel: a `train.jsonl` on a training machine has no repository attached,
and a label whose policy is unknown cannot be interpreted or compared
(§516). It sits outside `response` so it is never mistaken for something
the model should produce — the response contract is exactly
`{action, risk, explanation}`, and nothing else, in either direction.

## Files

| File | Rows | Purpose |
|---|---|---|
| `data/train.jsonl` | 8,000 | fine-tuning |
| `data/validation.jsonl` | 1,000 | fine-tuning checkpoints |
| `data/test.jsonl` | 1,000 | held-out measurement |
| `data/gold.jsonl` | 100 | stratified held-out set, 10 per category |
| `data/red_team.jsonl` | 26 | curated adversarial probes |

Measured on the written files, independently of the build's own report:
10,000 corpus rows, 10,000 unique `case_id`, 10,000 unique prompts,
**0** overlapping ids and **0** overlapping prompts across all ten pairs of
the five files.

## Labels

Every label is the deterministic engine's own verdict, produced by running
`make_recommendation` on the generated evidence (§65). Nothing in the
generator hardcodes an action or a risk. Re-derivable at any time by
replaying `recommend_for_context` over each row's context.

| Action | Rows | Share |
|---|---|---|
| `review` | 6,518 | 65.18% |
| `keep` | 3,482 | 34.82% |
| `delete` | **0** | 0% |

| Risk | Rows |
|---|---|
| `high` | 6,395 |
| `medium` | 2,675 |
| `low` | 930 |

`delete` is absent because `policy-v1` cannot produce it. A model fine-tuned
on this corpus has **never seen a `delete` label**, and any `delete` it emits
at serving time is an extrapolation the safety gate will reject.

## Coverage

All ten `FileCategory` values are present, and the build refuses to write
if any is missing:

| Category | Rows |
|---|---|
| `user_data` | 1,278 |
| `log` | 997 |
| `temporary` | 983 |
| `unknown` | 980 |
| `application_data` | 972 |
| `cache` | 964 |
| `driver` | 963 |
| `installer` | 961 |
| `crash_dump` | 951 |
| `system_data` | 951 |

Evidence spread:

| Property | Rows |
|---|---|
| `is_system_path` | 3,170 |
| `is_user_path` | 2,911 |
| `is_application_path` | 704 |
| `exists = false` | 654 |
| `exists = true`, `age_days = null` | 673 |
| at least one signal | 7,412 |
| no signals | 2,588 |

Signal counts, which together exercise the entire production vocabulary —
the build fails if the corpus uses a signal production cannot emit, or omits
one it can:

| Signal | Rows |
|---|---|
| `Located inside the Windows directory.` | 3,170 |
| `Located inside a user profile.` | 2,911 |
| `No deterministic classification rule matched.` | 980 |
| `Located inside Program Files.` | 704 |
| `File metadata could not be fully accessed.` | 673 |
| `File no longer exists.` | 654 |

## How the corpus is sampled

The unit of sampling is a **combination of evidence** — category ×
directory × extension × size × `exists` × `age_days` — not a random draw.
The full space is 13,020 combinations; 100 go to the gold set and 10,000 of
the remaining 12,920 form the corpus. Distinct semantic contexts therefore
equal row count by construction: **10,000 rows, 10,000 distinct contexts.**

- **Ages** (14 values plus a not-exists case) straddle the 30-day policy
  boundary from both sides at several distances: `null, 0, 1, 7, 28, 29,
  29.9, 30, 30.1, 31, 60, 180, 365, 1825`. 30 is the only number at which
  `policy-v1` changes its answer, so the corpus resolves it finely.
- **Sizes** span seven decades, 4 KB to 15 GB. `policy-v1` ignores size
  entirely, so this exists to contradict the most obvious wrong heuristic
  available to a model: "large therefore reclaimable."
- **Directories** are mixed within each category, so path-derived flags vary
  inside a category rather than being constant per category. Crash dumps
  appear under `C:\Windows\Minidump` *and* `C:\Users\Test\AppData\Local\CrashDumps`
  *and* `C:\Dumps`.

Flags and signals are derived by production code (`build_path_evidence`,
`apply_existence_evidence`), so a row's path and its flags cannot disagree,
and the signal wording is identical to what the scanner emits (§78).

## What this corpus deliberately does not contain

- **No `is_locked: true` rows.** Nothing in production sets `is_locked`
  (ADR 0002, Gap 1), so training on it would teach a feature the model is
  never shown at serving time. The previous corpus carried it on 2,691 rows
  with zero label signal. Locked cases live in `data/red_team.jsonl`
  instead.
- **No age on a missing file.** `exists = false` implies
  `age_days = null`, because a `stat` call on a file that is gone is what
  produces that `null`. The previous corpus gave missing files ages.
- **No `delete` labels**, per the table above.

## Honest limitations

These are properties of the corpus, not TODOs disguised as caveats. Each
one bounds what a model trained on it can be claimed to do.

1. **`data/gold.jsonl` has not been reviewed by a human.** Its labels are
   the engine's, exactly like the corpus. It is a *stratified held-out set*,
   and calling it a gold set in the §591 sense would be a false claim until
   someone actually reviews it. It is named `gold.jsonl` because that is
   where reviewed labels will land; the name is a destination, not a
   description.
2. **Only 12 distinct explanation strings across 10,000 rows.** They come
   from `policy-v1`'s twelve rules, one canned sentence each. A model
   fine-tuned on this will learn to select among twelve sentences, not to
   explain. Explanation quality cannot be measured against this corpus at
   all — only action and risk agreement can.
3. **Synthetic paths.** No real filesystem was scanned. The paths are
   plausible Windows locations with invented filenames; real machines have
   messier ones (spaces, Unicode, very long paths, junctions). §128
   long-path handling is untested by this corpus.
4. **Four categories carry no distinguishing label.** `cache`, `installer`,
   `driver`, and `system_data` all resolve to `review/medium` via
   `policy-v1`'s fallback rule. They are now *present* — they were absent
   entirely before — but they are indistinguishable from each other in the
   labels. ADR 0002, Gap 3.
5. **Class balance is a consequence, not a design.** 65/35 review/keep
   falls out of the evidence space crossed with the rule table. It was not
   targeted, and it is not a claim about real disk contents.
6. **Identical paths recur with different observed size and age.** This is
   intentional — the same file changes over time, and the model must key on
   evidence rather than on the path string — but it means row count is not
   a count of distinct files (§66).

## Red-team set

26 curated scenarios from `scenarios.py`, 12 of them written specifically
as conflicting evidence: cases where a plausible heuristic ("old therefore
stale", "large therefore reclaimable", "temporary therefore disposable")
gives the wrong answer. 7 carry `is_locked: true`, which is the one place
that flag appears. Labels: 15 `review` / 11 `keep`; 16 `high` / 5 `medium`
/ 5 `low`.

Disjoint from the corpus and from the gold set by build-time assertion, on
both `case_id` and rendered prompt.

## Regenerating

The corpus is a function of (`policy-v1`, the generator, seed 42). Any
change to the deterministic engine changes the labels and requires a full
regeneration plus re-measurement of every number on this page — see
"Changing this policy" in [`docs/policy-v1.md`](policy-v1.md). Metrics
measured against this corpus are not comparable across policy versions
(§514/§516).

The build refuses to write when any of these fail: empty corpus, duplicate
`case_id`, rows collapsing to fewer distinct contexts than rows, a missing
`FileCategory`, a signal outside the production vocabulary, an unexercised
signal, `current_action`/`current_risk` present in any prompt, or any
`case_id`/prompt shared between two of the five files. Each of those
assertions has a test that watches it fail on a crafted violation
(`tests/test_dataset_build.py`) — an assertion nobody has seen fail is
indistinguishable from one that cannot (§589).
