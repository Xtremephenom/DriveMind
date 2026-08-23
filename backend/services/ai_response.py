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
        "confidence",
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

    confidence = data["confidence"]

    if isinstance(confidence, bool) or not isinstance(
        confidence,
        (int, float),
    ):
        raise ValueError("AI confidence must be a number.")

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "AI confidence must be between 0 and 1."
        )

    explanation = data["explanation"]

    if not isinstance(explanation, str):
        raise ValueError("AI explanation must be a string.")

    return AIResponse(
        case_id=str(data.get("case_id", "")),
        action=action,
        risk=risk,
        confidence=confidence,
        explanation=explanation,
    )