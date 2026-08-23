from pathlib import Path

from backend.models.system import (
    FileCategory,
    FileSystemNode,
    NodeType,
)

from backend.services.classifier import classify_file


def make_file(path: str, size: int = 100) -> FileSystemNode:
    return FileSystemNode(
        path=path,
        node_type=NodeType.FILE,
        size=size,
        scanned=True,
    )


def test_windows_temp_file():
    node = make_file(
        r"C:\Windows\Temp\example.tmp",
        500,
    )

    result = classify_file(node)

    assert result.category == FileCategory.TEMPORARY
    assert result.extension == ".tmp"


def test_crash_dump_by_directory():
    node = make_file(
        r"C:\Windows\LiveKernelReports\example.dmp",
        5000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.CRASH_DUMP


def test_crash_dump_by_extension():
    node = make_file(
        r"C:\Somewhere\example.dmp",
        5000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.CRASH_DUMP


def test_chrome_cache():
    node = make_file(
        r"C:\Users\Test\AppData\Local\Google\Chrome\User Data\Default\Cache\data",
        1000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.CACHE


def test_windows_log():
    node = make_file(
        r"C:\Windows\Logs\example.log",
        1000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.LOG


def test_esupport_driver():
    node = make_file(
        r"C:\eSupport\eDriver\Software\Driver\NVIDIA\driver.msi",
        1000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.DRIVER


def test_user_data():
    node = make_file(
        r"C:\Users\Test\Documents\project.pdf",
        1000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.USER_DATA


def test_unknown_file():
    node = make_file(
        r"D:\Random\something.xyz",
        1000,
    )

    result = classify_file(node)

    assert result.category == FileCategory.UNKNOWN


def test_non_file_is_rejected():
    node = FileSystemNode(
        path=r"C:\Windows",
        node_type=NodeType.DIRECTORY,
        scanned=True,
    )

    try:
        classify_file(node)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")