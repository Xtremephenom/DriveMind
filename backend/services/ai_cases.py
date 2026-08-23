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
    """

    material = (
        f"{context.path}|"
        f"{context.size}|"
        f"{context.extension}|"
        f"{context.category.value}|"
        f"{context.age_days}|"
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