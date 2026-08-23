from __future__ import annotations

import json

from backend.models.system import AICase, ai_case_to_dict


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
10. Confidence must be a number between 0 and 1.

Return exactly this JSON structure:

{
  "action": "keep|review|delete",
  "risk": "low|medium|high",
  "confidence": 0.0,
  "explanation": "short explanation"
}
""".strip()


def build_ai_prompt(case: AICase) -> str:
    """
    Build the model input for a DriveMind AI case.

    The prompt contains only structured case information.
    """

    case_data = ai_case_to_dict(case)

    return (
        SYSTEM_INSTRUCTION
        + "\n\n"
        + "CASE:\n"
        + json.dumps(
            case_data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n\n"
        + "Analyze the case and return only the required JSON."
    )