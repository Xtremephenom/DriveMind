"""
End-to-end tests for the single AI path.

These are the tests that make §28 ("the deterministic engine is
authoritative") an observed property of the product rather than a
property of one unit-tested function. Every case here goes through the
real chain: engine -> context -> case -> provider -> parser -> gate.
"""

import pytest

from backend.models.system import (
    AICase,
    AIResponse,
    FileCategory,
    FileEvidence,
    FileRecord,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.ai import AIProvider
from backend.services.ai_response import parse_ai_response
from backend.services.ai_review import review_file
from backend.services.decision.engine import make_recommendation


def make_file(
    path=r"C:\Temp\old.tmp",
    category=FileCategory.TEMPORARY,
    size=1_000_000,
    age_days=90,
    is_system_path=False,
    is_user_path=False,
):
    extension = "." + path.rsplit(".", 1)[-1]

    file = FileRecord(
        path=path,
        size=size,
        category=category,
        extension=extension,
    )

    evidence = FileEvidence(
        path=path,
        size=size,
        extension=extension,
        age_days=age_days,
        exists=True,
        is_system_path=is_system_path,
        is_user_path=is_user_path,
        category=category,
        signals=["Old temporary file."],
    )

    return file, evidence


class TextProvider(AIProvider):
    """
    A provider that receives raw model text, exactly as a local LLM
    client would, and parses it with the production parser.
    """

    def __init__(self, text: str):
        self.text = text
        self.seen: list[AICase] = []

    def analyze(self, case: AICase) -> AIResponse:
        self.seen.append(case)

        parsed = parse_ai_response(self.text)

        return AIResponse(
            case_id=case.case_id,
            action=parsed.action,
            risk=parsed.risk,
            explanation=parsed.explanation,
        )


class RaisingProvider(AIProvider):
    def analyze(self, case: AICase) -> AIResponse:
        raise RuntimeError("The local model process died.")


class WrongCaseProvider(AIProvider):
    def analyze(self, case: AICase) -> AIResponse:
        return AIResponse(
            case_id="some-other-case",
            action=RecommendationAction.DELETE,
            risk=RecommendationRisk.LOW,
            explanation="Answering a different question.",
        )


class ContractProvider(AIProvider):
    """
    Returns exactly what the documented contract produces: the parser's
    output for `{action, risk, explanation}`, with no case id, because the
    prompt never asks for one.
    """

    def __init__(self, text: str):
        self.text = text

    def analyze(self, case: AICase) -> AIResponse:
        return parse_ai_response(self.text)


# --- No provider is the deterministic engine, exactly ------------------


def test_without_a_provider_the_result_is_the_deterministic_one():
    file, evidence = make_file()

    assert review_file(file, evidence) == make_recommendation(
        file, evidence
    )


# --- The gate is on the path -------------------------------------------


def test_provider_cannot_escalate_a_keep_case_to_delete():
    file, evidence = make_file(
        path=r"C:\Users\Test\Documents\thesis.pdf",
        category=FileCategory.USER_DATA,
        age_days=365,
        is_user_path=True,
    )

    provider = TextProvider(
        '{"action": "delete", "risk": "low",'
        ' "explanation": "Looks disposable."}'
    )

    result = review_file(file, evidence, provider)

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH
    assert "deterministic" in result.reason


def test_provider_cannot_lower_the_risk_floor():
    file, evidence = make_file(
        path=r"C:\Windows\Temp\system.etl",
        category=FileCategory.TEMPORARY,
        is_system_path=True,
    )

    provider = TextProvider(
        '{"action": "review", "risk": "low",'
        ' "explanation": "Harmless."}'
    )

    result = review_file(file, evidence, provider)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.HIGH


def test_a_safe_provider_answer_reaches_the_user():
    file, evidence = make_file()

    provider = TextProvider(
        '{"action": "review", "risk": "low",'
        ' "explanation": "An old temporary file, safe to review."}'
    )

    result = review_file(file, evidence, provider)

    assert result.action == RecommendationAction.REVIEW
    assert result.risk == RecommendationRisk.LOW
    assert result.reason == "An old temporary file, safe to review."

    # ...and the file's identity is never taken from the model.
    assert result.path == file.path
    assert result.size == file.size
    assert result.category == file.category


def test_a_more_conservative_provider_answer_is_kept():
    file, evidence = make_file()

    provider = TextProvider(
        '{"action": "keep", "risk": "high",'
        ' "explanation": "I would rather not."}'
    )

    result = review_file(file, evidence, provider)

    assert result.action == RecommendationAction.KEEP
    assert result.risk == RecommendationRisk.HIGH


# --- The prompt-facing case never carries the answer (§30/§31) ---------


def test_the_provider_is_never_asked_about_a_path_it_cannot_see():
    """
    The provider does receive the full context — leak prevention lives
    in `build_ai_prompt`, not here — but the case id must be content
    addressed, not the raw path.
    """

    file, evidence = make_file()
    provider = TextProvider(
        '{"action": "review", "risk": "low", "explanation": "ok"}'
    )

    review_file(file, evidence, provider)

    (case,) = provider.seen

    assert file.path not in case.case_id
    assert len(case.case_id) == 16


# --- Deterministic fallback (§544) ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "This is not JSON at all.",
        "",
        "{}",
        '{"action": "review", "risk": "low"}',
        '{"action": "obliterate", "risk": "low", "explanation": "x"}',
        '{"action": "review", "risk": "nuclear", "explanation": "x"}',
        '[{"action": "review", "risk": "low", "explanation": "x"}]',
    ],
)
def test_unparseable_provider_output_falls_back_to_the_engine(text):
    file, evidence = make_file()

    result = review_file(file, evidence, TextProvider(text))

    assert result == make_recommendation(file, evidence)


def test_a_crashing_provider_falls_back_to_the_engine():
    file, evidence = make_file()

    result = review_file(file, evidence, RaisingProvider())

    assert result == make_recommendation(file, evidence)


def test_an_answer_about_another_case_is_discarded():
    file, evidence = make_file()

    result = review_file(file, evidence, WrongCaseProvider())

    assert result == make_recommendation(file, evidence)
    assert result.action == RecommendationAction.REVIEW


def test_an_answer_with_no_case_id_is_accepted():
    """
    The regression this guards: requiring the provider to echo a case id
    rejects every response the documented contract can produce, because
    `build_ai_prompt` never shows the model an id to echo. The result was
    a path that silently never used the AI at all.
    """

    file, evidence = make_file()

    provider = ContractProvider(
        '{"action": "keep", "risk": "high",'
        ' "explanation": "Model advice, gate approved."}'
    )

    result = review_file(file, evidence, provider)

    assert result.action == RecommendationAction.KEEP
    assert result.reason == "Model advice, gate approved."
