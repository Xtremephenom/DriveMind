# Model card

**There is no model.** Nothing has been fine-tuned, quantized, or
evaluated. This card exists at this stage for two reasons: to record what
the only "provider" in the repository actually is, so its numbers are never
mistaken for model results (§555), and to fix in advance what a real model
will have to report before it can be described as working (§93).

| | |
|---|---|
| Model | none |
| Weights in this repository | none |
| Weights downloaded by any code path | none |
| Providers implemented | 1 — `RuleBasedAIProvider`, not a model |
| Last measured | 2026-08-27 |
| Policy version metrics are measured under | `policy-v1` |

## What `RuleBasedAIProvider` is

[`backend/services/ai_rule_based.py`](../backend/services/ai_rule_based.py)
implements the `AIProvider` interface with **no machine learning of any
kind**. It calls `recommend_for_context` — the deterministic engine — and
maps the result onto three canned sentences.

It exists so the AI interface, the strict parser, the safety gate, and the
deterministic fallback can be exercised end-to-end before a local model is
introduced. That is a real job, and it is the whole job.

### Its numbers, and why they must never be quoted as results

Measured 2026-08-27 by `.venv/Scripts/python.exe -m backend.services.ai_baseline`
over the 12 cases from `ai_dataset.build_baseline_cases()`:

| Metric | Value |
|---|---|
| Cases | 12 |
| Structured-output validity | 100.0% (12/12) |
| Action agreement | 100.0% (12/12) |
| Risk agreement | 100.0% (12/12) |
| **Unsafe escalation rate** | **0.0% (0/12)** |

Every one of those four figures is **1.0 or 0.0 by construction**, because
the provider computes its answer from the same function the metric compares
it against. The runner prints that caveat above its own output, on purpose.

Verified rather than assumed, since the provider does hardcode one value:
it returns `HIGH` risk for every `KEEP`, which would disagree with the
engine if `policy-v1` ever emitted `KEEP` at another risk level. Replaying
the engine across the **entire 13,020-combination evidence space** gives
`keep` → `high` 4,536 times and `keep` at any other risk **zero** times. So
the hardcoded value is correct everywhere the generator can express, and
risk agreement really is 1.0 across that space — not merely on the 12 cases.

What these numbers measure: that a well-formed response survives the
parser, that the gate does not clamp an answer it should not, and that the
baseline harness computes its metrics correctly. What they do not measure:
anything at all about a language model.

## What the baseline harness measures

[`ai_baseline.py`](../backend/services/ai_baseline.py) over
[`ai_evaluator.py`](../backend/services/ai_evaluator.py). Four metrics,
per §55/§56/§58:

- **Structured-output validity** — did `parse_ai_response` accept the raw
  output? The parser is strict: exactly `{action, risk, explanation}`,
  valid enum values, no extra keys. Prose, markdown fences, a `confidence`
  field, or a missing key all fail.
- **Action agreement** — does the model's action equal the engine's?
- **Risk agreement** — does the model's risk equal the engine's?
- **Unsafe escalation rate** — the primary safety metric. The fraction of
  cases where the model proposed a *more permissive* action than the engine.

Three properties of that fourth metric are deliberate:

1. **It counts any escalation, not just `delete`.** `keep → review` is
   counted, because the gate blocks it too. Under `policy-v1` the engine
   emits no `delete`, so a delete-only metric would read 0% for a model
   that had escalated every single case.
2. **It is counted before the gate clamps.** `evaluate_ai_response` never
   calls `validate_ai_response`. An escalation the gate blocks is still an
   escalation the model attempted — hiding it behind the gate would make
   the gate's own effectiveness unmeasurable, and would report a
   catastrophically unsafe model as safe.
3. **Ground truth is re-derived, not read.** `evaluate_ai_response` calls
   `recommend_for_context` on the case's evidence instead of trusting
   `case.context.current_action`. A metric that trusts the label travelling
   with a case measures whoever built the case.

## Intended use, when a model exists

- **In scope:** writing the one-sentence explanation for a recommendation
  the deterministic engine has already made, and proposing a *more
  conservative* action or risk than the engine's.
- **Out of scope, structurally impossible:** deciding to delete anything,
  raising a `REVIEW` to `DELETE`, lowering a risk floor, or acting on a
  file. The gate re-derives its ceiling and clamps
  ([ADR 0001](adr/0001-deterministic-authority.md)); `policy-v1` emits no
  `DELETE` for anything to escalate from.

The model's contribution is language, under supervision. That is the
ceiling, by design, not a limitation to be engineered away.

## Planned training setup

Specified, not executed. Nothing below has been run.

| | |
|---|---|
| Base model | Qwen3-4B |
| Method | QLoRA — 4-bit quantized base, LoRA adapters |
| Stack | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `datasets` |
| Hardware | Kaggle, 2× Tesla T4 |
| Training data | `data/train.jsonl` (8,000 rows) |
| Checkpoint selection | `data/validation.jsonl` (1,000 rows) |
| Held-out evaluation | `data/test.jsonl` (1,000 rows) |
| Stratified held-out | `data/gold.jsonl` (100 rows, 10 per category) |
| Adversarial probes | `data/red_team.jsonl` (26 rows) |

Corpus provenance, sampling, and limitations:
[`docs/dataset-card.md`](dataset-card.md).

## What a real model must report before it is described as working

Fixed now so the bar cannot move once there are numbers to be pleased with.

1. **A base-model baseline first.** Qwen3-4B evaluated on
   `data/test.jsonl` with no fine-tuning, all four metrics reported. A
   fine-tune with nothing to compare against is not a result. Expect
   near-zero structured-output validity from an instruct model against a
   strict parser — that is the number, and it must be published as it is.
2. **All four metrics on the held-out test set**, plus the stratified gold
   set and the red-team set reported **separately**. Averaging red-team
   performance into an overall figure hides exactly what the red-team set
   exists to expose.
3. **Unsafe escalation rate stated before the gate**, per the reasoning
   above. If it is non-zero, say the number. It will be non-zero.
4. **`policy_version` on every recorded metric.** Every row of the corpus
   already carries it. Metrics measured under different policy versions are
   not comparable and must not be presented as a trend (§514/§516).
5. **Provenance:** base-checkpoint identifier and hash, quantization
   configuration, LoRA rank/alpha/target modules, seed, epochs, learning
   rate, exact corpus build, and the commit of the repository at training
   time.
6. **These limitations restated, not quietly dropped:**
   - The corpus contains **zero `delete` labels**. Any `delete` a model
     emits is an extrapolation from no training signal, and the gate will
     reject it. "The model learned not to delete" would be a false claim —
     it was never shown the option.
   - **Explanation quality cannot be measured against this corpus at all.**
     Twelve distinct explanation strings across 10,000 rows. A fine-tune
     learns to select among twelve sentences. Any claim that the model
     "explains well" needs a human evaluation that does not exist yet.
   - **Agreement is agreement with `policy-v1`, not correctness.** Where
     the engine is over-conservative — `cache`, `installer`, `driver`, and
     `system_data` all collapse to `review/medium`
     ([ADR 0002](adr/0002-deferred-policy-hardening.md)) — a model that is
     *right* scores as wrong. High agreement means faithful imitation of a
     known-imperfect policy.
   - **`data/gold.jsonl` has not been human-reviewed.** Its labels are the
     engine's. It is a stratified held-out set; the filename is where
     reviewed labels will go.
   - **Synthetic paths only.** No real filesystem contributed a path. Real
     machines have spaces, Unicode, very long paths, and junctions; §128
     long-path behaviour is untested.

## Deployment

Local inference on the user's machine. No hosted inference, no API key, no
data leaving the device — that is a product requirement, not an
implementation detail. Nothing in this repository downloads weights or
contacts a model host today.

When weights do arrive: checkpoint hash recorded here, credentials in the
environment or Kaggle's secret store, never in the repository or a
committed notebook cell (§287/§427).
