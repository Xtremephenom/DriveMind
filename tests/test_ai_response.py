import pytest

from backend.models.system import (
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.ai_response import parse_ai_response


def test_valid_ai_response():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": 0.91,
        "explanation": "The file should be reviewed before cleanup."
    }
    """

    result = parse_ai_response(raw)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
    assert result.confidence == 0.91
    assert result.explanation == (
        "The file should be reviewed before cleanup."
    )


def test_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_ai_response("this is not json")


def test_missing_field():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": 0.9
    }
    """

    with pytest.raises(ValueError, match="missing required fields"):
        parse_ai_response(raw)


def test_invalid_action():
    raw = """
    {
        "action": "destroy",
        "risk": "low",
        "confidence": 0.9,
        "explanation": "Bad."
    }
    """

    with pytest.raises(ValueError, match="Invalid AI action"):
        parse_ai_response(raw)


def test_invalid_risk():
    raw = """
    {
        "action": "review",
        "risk": "banana",
        "confidence": 0.9,
        "explanation": "Bad."
    }
    """

    with pytest.raises(ValueError, match="Invalid AI risk"):
        parse_ai_response(raw)


def test_confidence_must_be_number():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": "very confident",
        "explanation": "Bad."
    }
    """

    with pytest.raises(ValueError, match="confidence must be a number"):
        parse_ai_response(raw)


def test_confidence_must_be_between_zero_and_one():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": 1.5,
        "explanation": "Bad."
    }
    """

    with pytest.raises(
        ValueError,
        match="confidence must be between 0 and 1",
    ):
        parse_ai_response(raw)


def test_explanation_must_be_string():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": 0.9,
        "explanation": 123
    }
    """

    with pytest.raises(
        ValueError,
        match="explanation must be a string",
    ):
        parse_ai_response(raw)