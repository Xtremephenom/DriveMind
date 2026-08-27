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
        "explanation": "The file should be reviewed before cleanup."
    }
    """

    result = parse_ai_response(raw)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
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
        "risk": "low"
    }
    """

    with pytest.raises(ValueError, match="missing required fields"):
        parse_ai_response(raw)


def test_invalid_action():
    raw = """
    {
        "action": "destroy",
        "risk": "low",
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
        "explanation": "Bad."
    }
    """

    with pytest.raises(ValueError, match="Invalid AI risk"):
        parse_ai_response(raw)


def test_confidence_is_not_part_of_the_contract():
    # A model that volunteers an uncalibrated confidence must still
    # parse: the field is ignored, never carried into the domain model.
    raw = """
    {
        "action": "review",
        "risk": "low",
        "confidence": 0.91,
        "explanation": "Review before cleanup."
    }
    """

    result = parse_ai_response(raw)

    assert result.action == RecommendationAction.REVIEW
    assert not hasattr(result, "confidence")


def test_explanation_must_be_string():
    raw = """
    {
        "action": "review",
        "risk": "low",
        "explanation": 123
    }
    """

    with pytest.raises(
        ValueError,
        match="explanation must be a string",
    ):
        parse_ai_response(raw)