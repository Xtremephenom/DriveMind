from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    JUNCTION = "junction"
    SKIPPED = "skipped"


@dataclass
class FileSystemNode:
    path: str
    node_type: NodeType
    size: int = 0
    scanned: bool = False
    reason: str | None = None
    children: list["FileSystemNode"] = field(default_factory=list)

@dataclass
class InventoryEntry:
    path: str
    size: int


@dataclass
class DriveInventory:
    path: str

    total_space: int
    free_space: int
    used_space: int

    scanned_size: int

    files: int
    directories: int
    junctions: int
    symlinks: int
    skipped: int

    largest_files: list[InventoryEntry] = field(default_factory=list)
    largest_directories: list[InventoryEntry] = field(default_factory=list)

class FileCategory(str, Enum):
    TEMPORARY = "temporary"
    CACHE = "cache"
    LOG = "log"
    CRASH_DUMP = "crash_dump"
    INSTALLER = "installer"
    DRIVER = "driver"
    USER_DATA = "user_data"
    APPLICATION_DATA = "application_data"
    SYSTEM_DATA = "system_data"
    UNKNOWN = "unknown"


@dataclass
class FileRecord:
    path: str
    size: int
    category: FileCategory
    extension: str | None = None
    reason: str | None = None

class RecommendationAction(str, Enum):
    DELETE = "delete"
    REVIEW = "review"
    KEEP = "keep"

class RecommendationRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Recommendation:
    path: str
    size: int
    category: FileCategory
    action: RecommendationAction
    risk: RecommendationRisk
    reason: str

@dataclass
class AnalysisResult:
    files: list[FileRecord] = field(default_factory=list)
    evidence: list[FileEvidence] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def review_size(self) -> int:
        """
        Bytes held by files a human should look at.

        This is **not** reclaimable space. `REVIEW` is an instruction to a
        person, not a verdict on the file, and a large share of these bytes
        will be kept once someone looks. Presenting this number as
        "space you can free" conflates action with risk (§23.7) and is a
        claim the engine has not made (§555).
        """

        return sum(
            recommendation.size
            for recommendation in self.recommendations
            if recommendation.action == RecommendationAction.REVIEW
        )

    @property
    def deletable_size(self) -> int:
        """
        Bytes the engine has actually recommended deleting.

        Currently always 0: `policy-v1` emits no `DELETE`, by design
        (`docs/policy-v1.md`). This property exists so that the honest
        answer to "how much can I free?" is a real measurement that
        happens to be zero, rather than `review_size` standing in for it.
        It starts reporting non-zero only when a policy that can produce
        `DELETE` ships, which ADR 0002 gates on closing every recorded
        policy gap.
        """

        return sum(
            recommendation.size
            for recommendation in self.recommendations
            if recommendation.action == RecommendationAction.DELETE
        )

@dataclass
class FileEvidence:
    path: str
    size: int
    extension: str

    age_days: float | None = None

    exists: bool = True
    is_locked: bool = False

    is_system_path: bool = False
    is_user_path: bool = False
    is_application_path: bool = False

    category: FileCategory = FileCategory.UNKNOWN

    signals: list[str] = field(default_factory=list)

@dataclass
class DecisionContext:
    path: str
    size: int
    extension: str

    category: FileCategory

    exists: bool
    age_days: float | None

    is_system_path: bool
    is_user_path: bool
    is_application_path: bool
    is_locked: bool

    signals: list[str] = field(default_factory=list)

    current_action: RecommendationAction = (
        RecommendationAction.KEEP
    )

    current_risk: RecommendationRisk = (
        RecommendationRisk.HIGH
    )

@dataclass
class AICase:
    case_id: str
    context: DecisionContext

@dataclass
class AIResponse:
    case_id: str
    action: RecommendationAction
    risk: RecommendationRisk
    explanation: str