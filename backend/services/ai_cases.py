from __future__ import annotations

import hashlib

from backend.models.system import (
    AICase,
    DecisionContext,
)


def build_case_id(context: DecisionContext) -> str:
    """
    Build a deterministic identifier for a filesystem case.

    The identifier does not contain the raw path.

    Every evidence field the prompt shows the model is hashed, and only
    those. `exists` is one of them: a file that is gone and a file that is
    there are different cases, so they must not share an id. They did
    until this was fixed, which meant a dataset containing both could
    report duplicate `case_id`s that were not duplicate cases.

    `current_action` / `current_risk` are deliberately excluded. The id
    identifies a *question*, not its answer, so re-labelling under a new
    policy version must not renumber the corpus.
    """

    material = (
        f"{context.path}|"
        f"{context.size}|"
        f"{context.extension}|"
        f"{context.category.value}|"
        f"{context.age_days}|"
        f"{context.exists}|"
        f"{context.is_system_path}|"
        f"{context.is_user_path}|"
        f"{context.is_application_path}|"
        f"{context.is_locked}"
    )

    return hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()[:16]


def build_ai_case(
    context: DecisionContext,
) -> AICase:
    return AICase(
        case_id=build_case_id(context),
        context=context,
    )