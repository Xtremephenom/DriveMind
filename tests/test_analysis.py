from backend.models.system import (
    FileCategory,
    FileSystemNode,
    NodeType,
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