from __future__ import annotations

from dataclasses import dataclass

from backend.models.system import FileCategory


@dataclass(frozen=True)
class Scenario:
    name: str
    category: FileCategory
    path: str
    extension: str
    age_days: float | None
    exists: bool
    is_locked: bool
    is_system_path: bool
    is_user_path: bool
    is_application_path: bool
    size: int


SCENARIOS = [
    # ---------------------------------------------------------
    # Temporary files
    # ---------------------------------------------------------

    Scenario(
        name="temp_recent",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=2,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=10 * 1024 * 1024,
    ),
    Scenario(
        name="temp_old",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=90,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="temp_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=90,
        exists=True,
        is_locked=True,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="temp_missing",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=90,
        exists=False,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=50 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Logs
    # ---------------------------------------------------------

    Scenario(
        name="log_recent",
        category=FileCategory.LOG,
        path=r"C:\Logs\file.log",
        extension=".log",
        age_days=3,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=20 * 1024 * 1024,
    ),
    Scenario(
        name="log_old",
        category=FileCategory.LOG,
        path=r"C:\Logs\file.log",
        extension=".log",
        age_days=90,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=100 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Crash dumps
    # ---------------------------------------------------------

    Scenario(
        name="dump_recent",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Dumps\file.dmp",
        extension=".dmp",
        age_days=2,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=100 * 1024 * 1024,
    ),
    Scenario(
        name="dump_old",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Dumps\file.dmp",
        extension=".dmp",
        age_days=120,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=200 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # User data
    # ---------------------------------------------------------

    Scenario(
        name="user_document",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\file.pdf",
        extension=".pdf",
        age_days=365,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=True,
        is_application_path=False,
        size=50 * 1024 * 1024,
    ),
    Scenario(
        name="user_photo",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Pictures\photo.jpg",
        extension=".jpg",
        age_days=365,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=True,
        is_application_path=False,
        size=10 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Application data
    # ---------------------------------------------------------

    Scenario(
        name="application_data",
        category=FileCategory.APPLICATION_DATA,
        path=r"C:\ProgramData\TestApp\data.bin",
        extension=".bin",
        age_days=90,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=True,
        size=100 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # System paths
    # ---------------------------------------------------------

    Scenario(
        name="system_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        extension=".log",
        age_days=180,
        exists=True,
        is_locked=False,
        is_system_path=True,
        is_user_path=False,
        is_application_path=False,
        size=200 * 1024 * 1024,
    ),
    Scenario(
        name="system_dump",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Windows\LiveKernelReports\memory.dmp",
        extension=".dmp",
        age_days=180,
        exists=True,
        is_locked=False,
        is_system_path=True,
        is_user_path=False,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    # ---------------------------------------------------------
    # Unknown
    # ---------------------------------------------------------

    Scenario(
        name="unknown",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\unknown.xyz",
        extension=".xyz",
        age_days=30,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=25 * 1024 * 1024,
    ),
        # ---------------------------------------------------------
    # Conflicting / boundary cases
    # ---------------------------------------------------------

    Scenario(
        name="temp_old_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=90,
        exists=True,
        is_locked=True,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="temp_recent_locked",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=2,
        exists=True,
        is_locked=True,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="temp_old_missing",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=120,
        exists=False,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="temp_recent_large",
        category=FileCategory.TEMPORARY,
        path=r"C:\Temp\file.tmp",
        extension=".tmp",
        age_days=1,
        exists=True,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=2 * 1024 * 1024 * 1024,
    ),

    Scenario(
        name="system_old_locked_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        extension=".log",
        age_days=365,
        exists=True,
        is_locked=True,
        is_system_path=True,
        is_user_path=False,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="system_recent_log",
        category=FileCategory.LOG,
        path=r"C:\Windows\System32\LogFiles\system.log",
        extension=".log",
        age_days=1,
        exists=True,
        is_locked=False,
        is_system_path=True,
        is_user_path=False,
        is_application_path=False,
        size=10 * 1024 * 1024,
    ),

    Scenario(
        name="system_old_dump_locked",
        category=FileCategory.CRASH_DUMP,
        path=r"C:\Windows\LiveKernelReports\memory.dmp",
        extension=".dmp",
        age_days=365,
        exists=True,
        is_locked=True,
        is_system_path=True,
        is_user_path=False,
        is_application_path=False,
        size=2 * 1024 * 1024 * 1024,
    ),

    Scenario(
        name="user_old_locked",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\important.pdf",
        extension=".pdf",
        age_days=3650,
        exists=True,
        is_locked=True,
        is_system_path=False,
        is_user_path=True,
        is_application_path=False,
        size=2 * 1024 * 1024 * 1024,
    ),

    Scenario(
        name="user_missing",
        category=FileCategory.USER_DATA,
        path=r"C:\Users\Test\Documents\missing.pdf",
        extension=".pdf",
        age_days=365,
        exists=False,
        is_locked=False,
        is_system_path=False,
        is_user_path=True,
        is_application_path=False,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="application_missing",
        category=FileCategory.APPLICATION_DATA,
        path=r"C:\ProgramData\TestApp\data.bin",
        extension=".bin",
        age_days=365,
        exists=False,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=True,
        size=500 * 1024 * 1024,
    ),

    Scenario(
        name="unknown_locked",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\unknown.xyz",
        extension=".xyz",
        age_days=365,
        exists=True,
        is_locked=True,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=2 * 1024 * 1024 * 1024,
    ),

    Scenario(
        name="unknown_missing",
        category=FileCategory.UNKNOWN,
        path=r"C:\somewhere\unknown.xyz",
        extension=".xyz",
        age_days=365,
        exists=False,
        is_locked=False,
        is_system_path=False,
        is_user_path=False,
        is_application_path=False,
        size=2 * 1024 * 1024 * 1024,
    ),
]