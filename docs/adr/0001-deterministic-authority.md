# ADR 0001 — The deterministic engine is the authority; the model is advisory

- **Status:** Accepted
- **Date:** 2026-08-27
- **Supersedes:** nothing
- **Related:** [ADR 0002](0002-deferred-policy-hardening.md),
  [`docs/policy-v1.md`](../policy-v1.md),
  [`docs/threat-model.md`](../threat-model.md)

## Context

DriveMind tells a user which files on their disk are safe to remove. A
wrong answer in the permissive direction destroys data that may not exist
anywhere else. There is no undo for a deleted file that was never backed
up.

DriveMind also uses a local language model, because the useful part of the
product is explaining *why* a file looks reclaimable, and that is a
language problem.

Those two facts are in tension. A language model is a probabilistic
component: it can be confidently wrong, it can be steered by its input,
and its failure modes are not enumerable in advance. A filename is input
the user does not control either — anything that writes a file to disk
chooses its own name, and that name reaches the model (§83/§316).

The question this ADR settles is: **what decides?**

## Decision

The deterministic engine decides. The model advises, and its advice is
only ever allowed to make an answer more conservative.

Concretely:

1. **The deterministic engine produces a complete recommendation before
   the model is consulted at all.** `make_recommendation(file, evidence)`
   in `backend/services/decision/engine.py` returns a full
   `Recommendation` from evidence alone. If no provider is configured,
   that recommendation *is* the product's answer. The AI path is an
   enhancement on top of a working product, not a dependency of it.

2. **The model receives evidence, never the verdict.** `build_ai_prompt`
   serializes the file's path, size, extension, category, existence, age,
   and path flags. It does not serialize `current_action` or
   `current_risk`, which travel on the same `DecisionContext` as ground
   truth for labelling. A model shown the answer cannot be evaluated
   against it.

3. **Every model answer passes through a gate that re-derives its own
   ceiling.** `validate_ai_response` calls `recommend_for_context`, which
   re-runs the engine on the case's evidence. The ceiling is *computed*,
   not read from the case. A gate whose limit arrives as an input is not a
   gate — it is a suggestion that the caller can raise.

4. **The gate is one-directional.** The model may propose `KEEP` where the
   engine said `REVIEW`, and `HIGH` risk where the engine said `LOW`. It
   may not propose `DELETE` where the engine said `REVIEW`, or `LOW` where
   the engine said `HIGH`. Those are silently replaced with the engine's
   verdict and an explanation saying so.

5. **Any model failure degrades to the deterministic answer.**
   `ai_review.review_file` wraps the provider call in a deliberately broad
   `except Exception`. A provider that raises, hangs and is killed, emits
   prose instead of JSON, emits JSON with an invalid enum value, or answers
   about a different case leaves the user with the engine's recommendation.
   There is no code path where a provider failure becomes a user-visible
   error or an absent recommendation.

6. **The only thing the model contributes that survives the gate is
   language.** In the best case the final `Recommendation.reason` is the
   model's `explanation`. The action and the risk are the engine's, or a
   more conservative value the model asked for.

## The chain, in one place

`backend/services/ai_review.py` is the only code path on which a model may
touch a recommendation:

```
FileRecord + FileEvidence
  -> make_recommendation        deterministic verdict, complete on its own
  -> build_decision_context     evidence + verdict travel together
  -> build_ai_case              content-addressed case id
  -> AIProvider.analyze         advisory answer (untrusted)
  -> validate_ai_response       gate re-derives the ceiling and clamps
  -> Recommendation             final
```

Every function in that chain already existed when this was wired; no new
abstraction was introduced to hold it (§464). That matters, because a
second path that skips a link is exactly how this property gets lost.

## Consequences

**Accepted, deliberately:**

- **The model cannot make DriveMind more useful in the permissive
  direction.** If the engine is over-conservative — and it is; four
  categories collapse to `REVIEW/MEDIUM`, see ADR 0002 — a perfectly
  correct model cannot fix that. The fix has to be made in the
  deterministic policy, reviewed, versioned, and published. This is
  slower. It is also the only version of "improvement" that is auditable.
- **Fine-tuning cannot raise the ceiling.** Training on
  `data/train.jsonl` teaches the model to *agree with* the engine.
  Agreement is the metric, so a model that is right where the engine is
  wrong scores as wrong. That is a known, accepted limitation of measuring
  against a deterministic label, recorded in
  [`docs/dataset-card.md`](../dataset-card.md).
- **Some of the safety architecture is currently unobservable in the
  product**, because `policy-v1` emits no `DELETE`, so the gate's most
  important clamp — refusing an escalation to `DELETE` — never fires on a
  real scan. It is exercised by tests with stub providers instead. This is
  why those tests exist and why they must not be deleted as redundant.

**Gained:**

- The product's safety is a property of code that does not learn. It does
  not change when the model is swapped, retrained, quantized differently,
  or prompt-injected by a filename.
- A prompt-injection attempt in a filename can, at absolute worst, make
  the model return an answer the gate then clamps. The attack surface for
  "make DriveMind recommend deleting this" is the deterministic policy, in
  the repository, under review — not the model's context window.
- Every recommendation is explainable without reference to model
  internals, because the action and the risk came from an enumerable rule
  table published in [`docs/policy-v1.md`](../policy-v1.md).

## Alternatives rejected

**Let the model decide, and use the engine as a sanity check on
egregious cases.** Rejected: this inverts the burden. Every case the
sanity check does not anticipate is decided by the model, and the set of
cases nobody anticipated is exactly where data loss lives.

**Weight the two and take a confidence-weighted blend.** Rejected on two
grounds. A blend of a deterministic verdict and a probabilistic one is
probabilistic. And it requires a calibrated confidence, which we do not
have — self-reported confidence from a language model is not evidence, and
an earlier iteration of this codebase asked the model for a `confidence`
field, made it mandatory in the parser, and had the rule-based provider
fabricate `0.99`. A fabricated number that looks like a measurement is
worse than no number.

**Skip the model entirely.** Tempting, and it would be a safe product.
Rejected because the engine's explanations are twelve canned sentences
from a rule table, and "here is why this 40 GB folder is safe to review"
in the user's own terms is most of the product's value. The model earns
its place by writing prose, under supervision, which is what point 6
above describes.

## How this is enforced

Not by convention. By tests that fail if it stops being true:

| Property | Test |
|---|---|
| No provider ⇒ engine's answer verbatim | `test_without_a_provider_the_result_is_the_deterministic_one` |
| Provider cannot escalate to `DELETE` | `test_provider_cannot_escalate_a_keep_case_to_delete` |
| Provider cannot lower the risk floor | `test_provider_cannot_lower_the_risk_floor` |
| A tampered `current_action` cannot widen the gate | `tests/test_ai_safety.py` |
| Malformed / non-JSON / crashing provider ⇒ engine's answer | `test_unparseable_provider_output_falls_back_to_the_engine`, `test_a_crashing_provider_falls_back_to_the_engine` |
| Verdict never reaches the prompt | `test_prompt_does_not_expose_current_decision`; and `build.check_no_label_leakage`, a build-time gate that refuses to write the corpus, watched by `test_label_leakage_is_rejected` |
| `DELETE` unreachable across the whole input space | `test_delete_is_unreachable` |

If a change makes one of those fail, the change is wrong, or it is a
deliberate policy revision that needs the full treatment in
[`docs/policy-v1.md`](../policy-v1.md)'s "Changing this policy" section.
Editing the expectation to match new behaviour is never the correct
response (§443).
