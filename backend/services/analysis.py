from __future__ import annotations

from backend.models.system import (
    AnalysisResult,
    FileSystemNode,
    NodeType,
)

from backend.services.classifier import classify_file

from backend.services.evidence import build_file_evidence
from backend.services.evidence import evidence_to_dict
from backend.services.decision.engine import make_recommendation


def analyze_tree(root: FileSystemNode) -> AnalysisResult:
    """
    Classify every accessible file in a filesystem tree and
    generate a recommendation for each classified file.

    This function is completely read-only.
    """

    result = AnalysisResult()

    _analyze_node(root, result)

    return result


def _analyze_node(
    node: FileSystemNode,
    result: AnalysisResult,
) -> None:

    if node.node_type == NodeType.FILE:
        file_record = classify_file(node)
        evidence = build_file_evidence(file_record)
        recommendation = make_recommendation(
            file_record,
            evidence,
        )

        result.files.append(file_record)
        result.evidence.append(evidence)
        result.recommendations.append(recommendation)

        return

    for child in node.children:
        _analyze_node(child, result)

def analysis_to_dict(
    result: AnalysisResult,
) -> dict:
    return {
        "total_files": result.total_files,
        # Two numbers, never one. `review_size` is attention required;
        # `deletable_size` is space actually reclaimable and is 0 under
        # policy-v1. Summing them, or labelling either "recommended",
        # would present bytes needing a human decision as bytes the user
        # can free (§23.7).
        "review_size": result.review_size,
        "deletable_size": result.deletable_size,
        "files": [
            {
                "path": file.path,
                "size": file.size,
                "category": file.category.value,
                "extension": file.extension,
                "reason": file.reason,
            }
            for file in result.files
        ],
        "recommendations": [
            {
                "path": recommendation.path,
                "size": recommendation.size,
                "category": recommendation.category.value,
                "action": recommendation.action.value,
                "risk": recommendation.risk.value,
                "reason": recommendation.reason,
            }
            for recommendation in result.recommendations
        ],
        "evidence": [
            evidence_to_dict(evidence)
            for evidence in result.evidence
        ],
    }