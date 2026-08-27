from backend.models.system import (
    AnalysisResult,
    FileCategory,
    FileSystemNode,
    NodeType,
    Recommendation,
    RecommendationAction,
    RecommendationRisk,
)

from backend.services.analysis import (
    analyze_tree,
    analysis_to_dict,
)


def make_file(
    path: str,
    size: int,
) -> FileSystemNode:
    return FileSystemNode(
        path=path,
        node_type=NodeType.FILE,
        size=size,
        scanned=True,
    )


def make_directory(
    path: str,
    children: list[FileSystemNode],
) -> FileSystemNode:
    return FileSystemNode(
        path=path,
        node_type=NodeType.DIRECTORY,
        scanned=True,
        children=children,
    )


def test_analyze_single_file():
    file = make_file(
        r"C:\Windows\Temp\test.tmp",
        100,
    )

    result = analyze_tree(file)

    assert result.total_files == 1
    assert len(result.files) == 1
    assert len(result.recommendations) == 1

    assert result.files[0].category == FileCategory.TEMPORARY


def test_analyze_nested_tree():
    file1 = make_file(
        r"C:\Windows\Temp\a.tmp",
        100,
    )

    file2 = make_file(
        r"C:\Windows\LiveKernelReports\b.dmp",
        200,
    )

    subdirectory = make_directory(
        r"C:\Windows\LiveKernelReports",
        [file2],
    )

    root = make_directory(
        r"C:\Windows",
        [file1, subdirectory],
    )

    result = analyze_tree(root)

    assert result.total_files == 2
    assert len(result.files) == 2
    assert len(result.recommendations) == 2


def test_empty_tree():
    root = make_directory(
        r"C:\Empty",
        [],
    )

    result = analyze_tree(root)

    assert result.total_files == 0
    assert result.files == []
    assert result.recommendations == []


def test_analysis_to_dict():
    file = make_file(
        r"C:\Windows\LiveKernelReports\test.dmp",
        5000,
    )

    result = analyze_tree(file)
    data = analysis_to_dict(result)

    assert data["total_files"] == 1
    assert len(data["files"]) == 1
    assert len(data["recommendations"]) == 1

    assert data["files"][0]["category"] == "crash_dump"
    assert data["recommendations"][0]["action"] == "review"


def test_nested_analysis_preserves_file_sizes():
    file1 = make_file(
        r"C:\Windows\Temp\a.tmp",
        1000,
    )

    file2 = make_file(
        r"C:\Users\Test\Documents\b.pdf",
        2000,
    )

    root = make_directory(
        r"C:\Test",
        [file1, file2],
    )

    result = analyze_tree(root)

    assert sum(file.size for file in result.files) == 3000

def test_analysis_contains_evidence():
    file = make_file(
        r"C:\Windows\Temp\test.tmp",
        1000,
    )

    result = analyze_tree(file)

    assert len(result.evidence) == 1

    evidence = result.evidence[0]

    assert evidence.path == file.path
    assert evidence.size == 1000
    assert evidence.extension == ".tmp"
    assert evidence.is_system_path is True


# --- Reported sizes separate attention from reclaimable space (§23.7) ---


def test_review_bytes_are_not_reported_as_reclaimable():
    """
    The regression this guards: a single `total_recommended_size` summing
    `REVIEW` items, surfaced under a name the user reads as "space you can
    free". Every byte here needs a human decision; none of it is freed.
    """

    root = make_directory(
        r"C:\Test",
        [
            make_file(r"C:\Windows\Temp\a.tmp", 1000),
            make_file(r"C:\Users\Test\Documents\b.pdf", 2000),
        ],
    )

    result = analyze_tree(root)

    actions = {r.action for r in result.recommendations}

    assert RecommendationAction.DELETE not in actions

    assert result.review_size == 1000
    assert result.deletable_size == 0

    # The two are independent measurements, not a split of one total.
    assert not hasattr(result, "total_recommended_size")


def test_deletable_size_counts_only_delete():
    """
    `deletable_size` is 0 today because `policy-v1` emits no `DELETE`, so
    the property is asserted against a hand-built `DELETE` recommendation.
    A number that is only ever observed as zero is not a measurement.
    """

    result = AnalysisResult(
        recommendations=[
            Recommendation(
                path=r"C:\Temp\gone.tmp",
                size=4096,
                category=FileCategory.TEMPORARY,
                action=RecommendationAction.DELETE,
                risk=RecommendationRisk.LOW,
                reason="Hypothetical future policy.",
            ),
            Recommendation(
                path=r"C:\Temp\keep.tmp",
                size=8192,
                category=FileCategory.TEMPORARY,
                action=RecommendationAction.REVIEW,
                risk=RecommendationRisk.LOW,
                reason="Needs a look.",
            ),
        ]
    )

    assert result.deletable_size == 4096
    assert result.review_size == 8192


def test_serialized_analysis_reports_both_sizes_separately():
    file = make_file(r"C:\Windows\Temp\test.tmp", 5000)

    data = analysis_to_dict(analyze_tree(file))

    assert data["review_size"] == 5000
    assert data["deletable_size"] == 0
    assert "total_recommended_size" not in data
