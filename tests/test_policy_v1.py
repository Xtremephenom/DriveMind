"""
Characterization tests for `policy-v1`.

`docs/policy-v1.md` publishes a truth table, and every label in
`data/*.jsonl` was produced by the engine this table describes. These
tests re-measure the table on every run so the document cannot drift
away from the code, and so an accidental policy change fails the suite
rather than silently invalidating the dataset (§514).

If a test here fails, one of two things is true: the policy changed and
`docs/policy-v1.md` plus the dataset need the full treatment in that
document's "Changing this policy" section, or the change was accidental
and should be reverted. Editing the expectations to match new behaviour
is the one response that is never correct.
"""

import pytest

from backend.models.system import (
    FileCategory,
    FileEvidence,
    FileRecord,
    RecommendationAction,
    RecommendationRisk,
)
from backend.services.decision.engine import (
    POLICY_VERSION,
    make_recommendation,
)


def verdict(
    category,
    *,
    is_system_path=False,
    exists=True,
    age_days=None,
    is_user_path=False,
    is_application_path=False,
    is_locked=False,
    size=1234,
    signals=None,
):
    """Run one input combination through the engine."""

    file = FileRecord(
        path=r"C:\somewhere\subject.dat",
        size=size,
        category=category,
        extension=".dat",
    )

    evidence = FileEvidence(
        path=file.path,
        size=size,
        extension=".dat",
        age_days=age_days,
        exists=exists,
        is_locked=is_locked,
        is_system_path=is_system_path,
        is_user_path=is_user_path,
        is_application_path=is_application_path,
        category=category,
        signals=list(signals or []),
    )

    recommendation = make_recommendation(file, evidence)

    return recommendation.action, recommendation.risk


KEEP_HIGH = (RecommendationAction.KEEP, RecommendationRisk.HIGH)
REVIEW_HIGH = (RecommendationAction.REVIEW, RecommendationRisk.HIGH)
REVIEW_MEDIUM = (RecommendationAction.REVIEW, RecommendationRisk.MEDIUM)
REVIEW_LOW = (RecommendationAction.REVIEW, RecommendationRisk.LOW)

AGES = (None, 0, 29, 30, 400)

# The truth table published in docs/policy-v1.md, transcribed as
# {category: {(is_system_path, exists): {age: verdict}}}.
AGE_SENSITIVE = {
    None: REVIEW_MEDIUM,
    0: REVIEW_MEDIUM,
    29: REVIEW_MEDIUM,
    30: REVIEW_LOW,
    400: REVIEW_LOW,
}

AGE_BLIND_MEDIUM = dict.fromkeys(AGES, REVIEW_MEDIUM)
ALWAYS_KEEP_HIGH = dict.fromkeys(AGES, KEEP_HIGH)
ALWAYS_REVIEW_HIGH = dict.fromkeys(AGES, REVIEW_HIGH)

TRUTH_TABLE = {
    FileCategory.TEMPORARY: {
        (False, True): AGE_SENSITIVE,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.LOG: {
        (False, True): AGE_SENSITIVE,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.CRASH_DUMP: {
        (False, True): AGE_SENSITIVE,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.CACHE: {
        (False, True): AGE_BLIND_MEDIUM,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.INSTALLER: {
        (False, True): AGE_BLIND_MEDIUM,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.DRIVER: {
        (False, True): AGE_BLIND_MEDIUM,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.SYSTEM_DATA: {
        (False, True): AGE_BLIND_MEDIUM,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_REVIEW_HIGH,
        (True, False): ALWAYS_REVIEW_HIGH,
    },
    FileCategory.USER_DATA: {
        (False, True): ALWAYS_KEEP_HIGH,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_KEEP_HIGH,
        (True, False): ALWAYS_KEEP_HIGH,
    },
    FileCategory.APPLICATION_DATA: {
        (False, True): ALWAYS_KEEP_HIGH,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_KEEP_HIGH,
        (True, False): ALWAYS_KEEP_HIGH,
    },
    FileCategory.UNKNOWN: {
        (False, True): ALWAYS_KEEP_HIGH,
        (False, False): ALWAYS_KEEP_HIGH,
        (True, True): ALWAYS_KEEP_HIGH,
        (True, False): ALWAYS_KEEP_HIGH,
    },
}


def test_the_table_covers_every_category():
    """A new FileCategory must be given a documented verdict."""

    assert set(TRUTH_TABLE) == set(FileCategory)


@pytest.mark.parametrize("category", list(FileCategory))
@pytest.mark.parametrize("is_system_path", [False, True])
@pytest.mark.parametrize("exists", [True, False])
@pytest.mark.parametrize("age_days", AGES)
def test_engine_matches_the_published_truth_table(
    category,
    is_system_path,
    exists,
    age_days,
):
    expected = TRUTH_TABLE[category][(is_system_path, exists)][age_days]

    assert (
        verdict(
            category,
            is_system_path=is_system_path,
            exists=exists,
            age_days=age_days,
        )
        == expected
    )


# --- Properties the document promises ---------------------------------


@pytest.mark.parametrize("category", list(FileCategory))
def test_delete_is_unreachable(category):
    """
    Property 1. Every gap in ADR 0002 is latent only because of this,
    so it is asserted over the whole input space rather than assumed.
    """

    for is_system_path in (False, True):
        for exists in (True, False):
            for age_days in AGES:
                for is_locked in (False, True):
                    action, _ = verdict(
                        category,
                        is_system_path=is_system_path,
                        exists=exists,
                        age_days=age_days,
                        is_locked=is_locked,
                    )

                    assert action != RecommendationAction.DELETE


@pytest.mark.parametrize(
    "category",
    [
        FileCategory.USER_DATA,
        FileCategory.APPLICATION_DATA,
        FileCategory.UNKNOWN,
    ],
)
def test_irrecoverable_categories_are_unconditional(category):
    """Property 3: nothing moves these off keep/high."""

    for is_system_path in (False, True):
        for exists in (True, False):
            for age_days in AGES:
                assert (
                    verdict(
                        category,
                        is_system_path=is_system_path,
                        exists=exists,
                        age_days=age_days,
                    )
                    == KEEP_HIGH
                )


def test_the_thirty_day_boundary_is_inclusive():
    assert verdict(
        FileCategory.TEMPORARY, age_days=29.999
    ) == REVIEW_MEDIUM
    assert verdict(FileCategory.TEMPORARY, age_days=30) == REVIEW_LOW


def test_unknown_age_is_treated_as_recent():
    """The conservative direction, per the rule table's note."""

    assert verdict(FileCategory.LOG, age_days=None) == REVIEW_MEDIUM


# --- Fields the engine does not read ----------------------------------


@pytest.mark.parametrize(
    "ignored",
    [
        {"is_locked": True},
        {"is_user_path": True},
        {"is_application_path": True},
        {"signals": ["Something alarming."]},
        {"size": 40_000_000_000},
        {"size": 0},
    ],
)
def test_ignored_fields_do_not_change_the_verdict(ignored):
    """
    Documented in policy-v1.md's "What the engine reads". `is_locked`
    appearing here is the whole of ADR 0002 Gap 1: the field exists on
    the context, nothing writes it, and it moves nothing. That is why
    the training corpus excludes it rather than carrying it as noise.
    """

    baseline = verdict(FileCategory.TEMPORARY, age_days=90)

    assert verdict(
        FileCategory.TEMPORARY, age_days=90, **ignored
    ) == baseline


def test_size_never_influences_the_verdict():
    """
    A 40 GB log and a 4 KB log get the same answer. Worth asserting
    because "biggest first" is the obvious next feature request and it
    must not become a policy input by accident.
    """

    small = verdict(FileCategory.LOG, age_days=400, size=4_096)
    huge = verdict(FileCategory.LOG, age_days=400, size=40_000_000_000)

    assert small == huge


# --- The documented ordering artifact ---------------------------------


def test_missing_file_in_a_system_path_is_reviewed_not_kept():
    """
    Recorded in policy-v1.md under "Known artifacts of the ordering":
    rule 4 precedes rule 5, so a file that does not exist still gets
    review/high if its path looks like a system path. Harmless while
    `delete` is unreachable; asserted so the artifact is visible rather
    than a surprise.
    """

    assert verdict(
        FileCategory.TEMPORARY,
        is_system_path=True,
        exists=False,
    ) == REVIEW_HIGH

    assert verdict(
        FileCategory.TEMPORARY,
        is_system_path=False,
        exists=False,
    ) == KEEP_HIGH


# --- The version is a real artifact, not prose ------------------------


def test_the_policy_version_names_the_published_document():
    """
    `POLICY_VERSION` is what stamps every dataset row and every recorded
    metric. If it names a document that does not exist, the provenance it
    provides is fictional, so the constant and the file are checked
    against each other rather than each being trusted alone (§514/§516).
    """

    from pathlib import Path

    document = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / f"{POLICY_VERSION}.md"
    )

    assert document.is_file(), (
        f"POLICY_VERSION is {POLICY_VERSION!r} but "
        f"docs/{POLICY_VERSION}.md does not exist"
    )

    # The tests in this module measure the policy; this asserts they are
    # measuring the one the document claims to describe.
    assert POLICY_VERSION in document.read_text(encoding="utf-8")
