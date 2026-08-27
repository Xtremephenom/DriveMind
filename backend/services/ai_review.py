"""
The one path on which the AI is allowed to touch a recommendation.

The chain is fixed and every link already existed:

    FileRecord + FileEvidence
      -> make_recommendation        (deterministic verdict)
      -> build_decision_context     (evidence + verdict, no leak into the prompt)
      -> build_ai_case              (content-addressed case id)
      -> AIProvider.analyze         (advisory answer)
      -> validate_ai_response       (deterministic safety gate)
      -> Recommendation             (final)

Two properties this module is responsible for:

*   **The AI can only narrow.** The gate is not optional and not
    bypassable — there is no branch here that returns a model answer
    without passing it through `validate_ai_response` (§28/§443).
*   **The AI can never break the product.** A provider that raises, times
    out, emits unparseable text, or answers about a different case leaves
    the user with the deterministic recommendation, which was complete on
    its own before the provider was ever called (§544).
"""

from __future__ import annotations

from backend.models.system import (
    FileEvidence,
    FileRecord,
    Recommendation,
)
from backend.services.ai import AIProvider
from backend.services.ai_cases import build_ai_case
from backend.services.ai_safety.validator import validate_ai_response
from backend.services.context import build_decision_context
from backend.services.decision.engine import make_recommendation


def review_file(
    file: FileRecord,
    evidence: FileEvidence,
    provider: AIProvider | None = None,
) -> Recommendation:
    """
    Produce the final recommendation for one file.

    With no provider this is exactly `make_recommendation`. With a
    provider the answer may be made more conservative, and its
    explanation may replace the deterministic reason — nothing else.
    """

    deterministic = make_recommendation(file, evidence)

    if provider is None:
        return deterministic

    context = build_decision_context(file, evidence, deterministic)
    case = build_ai_case(context)

    try:
        response = provider.analyze(case)

        # The prompt does not ask for a case id, so a well-behaved model
        # returns none and `parse_ai_response` leaves it empty. A provider
        # that volunteers one and gets it wrong is answering about a
        # different case: treat that as a failure rather than as advice.
        # Requiring the echo unconditionally would reject every response
        # the documented contract produces.
        if response.case_id and response.case_id != case.case_id:
            return deterministic

        gated = validate_ai_response(case, response)

    except Exception:
        # Deliberately broad: a provider is an out-of-process model or a
        # third-party client, and no failure of it may propagate into a
        # scan. Diagnostics belong to the provider, which is the only
        # layer that knows what actually failed.
        return deterministic

    return Recommendation(
        path=deterministic.path,
        size=deterministic.size,
        category=deterministic.category,
        action=gated.action,
        risk=gated.risk,
        reason=gated.explanation,
    )
