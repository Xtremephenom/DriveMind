from __future__ import annotations

import json

from backend.models.system import AICase


SYSTEM_INSTRUCTION = """
You are DriveMind, a local AI assistant for analyzing disk-space usage.

Your job is to reason about a filesystem case and provide a conservative
recommendation.

Safety rules:

1. Never assume a file is safe to delete merely because it is large.
2. Never recommend deletion of user data without strong evidence.
3. Treat Windows system paths as high risk.
4. When evidence is insufficient, choose "keep" or "review".
5. Never invent facts that are not present in the case.
6. Never execute commands or modify the filesystem.
7. Your output must be valid JSON.
8. The action must be exactly one of:
   "keep", "review", "delete".
9. The risk must be exactly one of:
   "low", "medium", "high".

Return exactly this JSON structure:

{
  "action": "keep|review|delete",
  "risk": "low|medium|high",
  "explanation": "short explanation"
}
""".strip()


def build_ai_prompt(case: AICase) -> str:
    """
    Build the model-facing prompt for a DriveMind case.

    The model receives filesystem evidence only.
    Deterministic DriveMind decisions are intentionally hidden
    from the model because they are used as ground-truth labels.
    """

    context_dict = {
        "path": case.context.path,
        "size": case.context.size,
        "extension": case.context.extension,
        "category": case.context.category.value,
        "exists": case.context.exists,
        "age_days": case.context.age_days,
        "is_system_path": case.context.is_system_path,
        "is_user_path": case.context.is_user_path,
        "is_application_path": case.context.is_application_path,
        "is_locked": case.context.is_locked,
        "signals": case.context.signals,
    }

    case_data = {
        "case_id": case.case_id,
        "context": context_dict,
    }

    case_json = json.dumps(
        case_data,
        indent=2,
        ensure_ascii=False,
    )

    return (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + "CASE:\n"
        + case_json
        + "\n\n"
        + "Analyze the case and return only the required JSON."
    )