from __future__ import annotations

import json

from backend.models.system import (
    AIResponse,
    RecommendationAction,
    RecommendationRisk,
)


def parse_ai_response(raw_text: str) -> AIResponse:
    """
    Parse and validate raw model output.

    The model output is untrusted input.
    """

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("AI response is not valid JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("AI response must be a JSON object.")

    required = {
        "action",
        "risk",
        "explanation",
    }

    missing = required - data.keys()

    if missing:
        raise ValueError(
            f"AI response is missing required fields: "
            f"{', '.join(sorted(missing))}."
        )

    try:
        action = RecommendationAction(data["action"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid AI action: {data['action']!r}."
        ) from exc

    try:
        risk = RecommendationRisk(data["risk"])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid AI risk: {data['risk']!r}."
        ) from exc

    # A confidence field is deliberately NOT part of the contract.
    # An uncalibrated self-reported confidence is not evidence, so it
    # is neither required nor carried into the domain model. If a model
    # volunteers one it is ignored rather than treated as a signal.

    explanation = data["explanation"]

    if not isinstance(explanation, str):
        raise ValueError("AI explanation must be a string.")

    return AIResponse(
        case_id=str(data.get("case_id", "")),
        action=action,
        risk=risk,
        explanation=explanation,
    )