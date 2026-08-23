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

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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
    def total_recommended_size(self) -> int:
        return sum(
            recommendation.size
            for recommendation in self.recommendations
            if recommendation.action == RecommendationAction.REVIEW
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

def ai_case_to_dict(case: AICase) -> dict:
    return {
        "case_id": case.case_id,
        "context": {
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
            "current_action": case.context.current_action.value,
            "current_risk": case.context.current_risk.value,
        },
    }

@dataclass
class AIResponse:
    case_id: str
    action: RecommendationAction
    risk: RecommendationRisk
    confidence: float
    explanation: str