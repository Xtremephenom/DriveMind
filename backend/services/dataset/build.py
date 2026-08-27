"""
The dataset build.

`build_dataset` is a checkpoint, not just a writer. Every property the
repository claims about `data/*.jsonl` -- category coverage, distinct
contexts, no duplicate ids, no cross-split leakage, no label leakage into
prompts -- is asserted here *before* anything is written, and a violation
raises instead of producing a file.

The reason is that a bad dataset is silent. Nothing downstream fails; the
fine-tune simply learns the wrong thing and the eval numbers look
plausible. The previous corpus shipped with 10,000 rows that were really
512 distinct cases, four of ten categories missing, and `signals` empty
throughout, and none of that surfaced until someone counted (§589).
"""

from __future__ import annotations

from collections import Counter

from backend.models.system import AICase, FileCategory
from backend.services.ai_prompt import build_ai_prompt
from backend.services.dataset.generator import generate_corpus_and_gold
from backend.services.dataset.labeler import label_case
from backend.services.dataset.scenarios import generate_red_team_cases
from backend.services.dataset.split import split_cases
from backend.services.dataset.writer import write_jsonl
from backend.services.decision.engine import POLICY_VERSION
from backend.services.evidence import SIGNAL_VOCABULARY


class DatasetIntegrityError(AssertionError):
    """A build-time invariant was violated. Nothing has been written."""


def _semantic_key(case: AICase) -> tuple:
    """
    Everything the model is shown about a case.

    Two cases with the same key ask the same question, whatever their
    `case_id` says. This is what "distinct semantic contexts" counts, and
    it is deliberately independent of `case_id` so a change to the id
    scheme cannot hide a duplication.
    """

    context = case.context

    return (
        context.path,
        context.size,
        context.extension,
        context.category,
        context.exists,
        context.age_days,
        context.is_system_path,
        context.is_user_path,
        context.is_application_path,
        context.is_locked,
        tuple(context.signals),
    )


def check_corpus(cases: list[AICase]) -> None:
    """Invariants that must hold over the whole corpus."""

    if not cases:
        raise DatasetIntegrityError("the corpus is empty")

    ids = [case.case_id for case in cases]

    if len(set(ids)) != len(ids):
        duplicates = [
            case_id
            for case_id, count in Counter(ids).items()
            if count > 1
        ]

        raise DatasetIntegrityError(
            f"{len(duplicates)} duplicate case_id(s), "
            f"e.g. {duplicates[:3]}"
        )

    keys = {_semantic_key(case) for case in cases}

    if len(keys) != len(cases):
        raise DatasetIntegrityError(
            f"{len(cases)} rows collapse to {len(keys)} distinct "
            "semantic contexts; the corpus is padded, not diverse"
        )

    present = {case.context.category for case in cases}
    missing = set(FileCategory) - present

    if missing:
        raise DatasetIntegrityError(
            "categories absent from the corpus: "
            f"{sorted(category.value for category in missing)}"
        )

    observed = {
        signal
        for case in cases
        for signal in case.context.signals
    }

    invented = observed - SIGNAL_VOCABULARY

    if invented:
        raise DatasetIntegrityError(
            f"corpus contains signals production cannot emit: "
            f"{sorted(invented)}"
        )

    unexercised = SIGNAL_VOCABULARY - observed

    if unexercised:
        raise DatasetIntegrityError(
            "signals production can emit but the corpus never does: "
            f"{sorted(unexercised)}"
        )


def check_no_label_leakage(cases: list[AICase]) -> None:
    """
    No prompt may contain the deterministic verdict (§30/§31).

    Checked on the rendered prompt rather than on the serializer, because
    the serializer is what would have to be wrong for this to fail, and a
    test of a serializer against itself proves nothing.
    """

    for case in cases:
        prompt = build_ai_prompt(case)

        for forbidden in ("current_action", "current_risk"):
            if forbidden in prompt:
                raise DatasetIntegrityError(
                    f"case {case.case_id} leaks {forbidden} into its "
                    "prompt"
                )


def check_disjoint(
    named_sets: dict[str, list[AICase]],
) -> None:
    """
    No case_id and no rendered prompt may appear in two sets.

    Prompts are compared as well as ids because an id collision is the
    obvious leak and a prompt collision is the one that survives an id
    scheme change.
    """

    names = list(named_sets)

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared_ids = {
                case.case_id for case in named_sets[left]
            } & {
                case.case_id for case in named_sets[right]
            }

            if shared_ids:
                raise DatasetIntegrityError(
                    f"{len(shared_ids)} case_id(s) shared between "
                    f"{left} and {right}"
                )

            shared_prompts = {
                build_ai_prompt(case) for case in named_sets[left]
            } & {
                build_ai_prompt(case) for case in named_sets[right]
            }

            if shared_prompts:
                raise DatasetIntegrityError(
                    f"{len(shared_prompts)} prompt(s) shared between "
                    f"{left} and {right}"
                )


def build_dataset(
    count: int = 10_000,
    *,
    seed: int = 42,
    gold_per_category: int = 10,
) -> dict:
    corpus, gold = generate_corpus_and_gold(
        count,
        gold_per_category=gold_per_category,
        seed=seed,
    )

    red_team = generate_red_team_cases()

    check_corpus(corpus)
    check_no_label_leakage(corpus + gold + red_team)

    split = split_cases(corpus, seed=seed)

    check_disjoint(
        {
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
            "gold": gold,
            "red_team": red_team,
        }
    )

    actions = Counter()
    risks = Counter()
    categories = Counter()

    for case in corpus:
        recommendation = label_case(case)

        actions[recommendation.action.value] += 1
        risks[recommendation.risk.value] += 1
        categories[case.context.category.value] += 1

    write_jsonl(split.train, "data/train.jsonl")
    write_jsonl(split.validation, "data/validation.jsonl")
    write_jsonl(split.test, "data/test.jsonl")
    write_jsonl(gold, "data/gold.jsonl")
    write_jsonl(red_team, "data/red_team.jsonl")

    return {
        "total": len(corpus),
        "distinct_contexts": len(
            {_semantic_key(case) for case in corpus}
        ),
        "actions": dict(actions),
        "risks": dict(risks),
        "categories": dict(categories),
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
        "gold": len(gold),
        "red_team": len(red_team),
        "seed": seed,
        # The two things that make the numbers above interpretable later.
        # Without them a recorded distribution is a measurement of an
        # unnamed policy over an unreproducible sample (§514/§516).
        "policy_version": POLICY_VERSION,
    }
