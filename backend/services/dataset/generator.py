"""
The one generator for DriveMind's synthetic training corpus.

Four properties distinguish it from the two generators it replaces:

*   **Every case is semantically distinct.** The unit sampled is a
    *combination of evidence*, not a random draw with a row index glued
    into the filename. The generator it replaced produced 10,000 rows that
    reduced to 512 distinct semantic contexts, one of them repeated 49
    times, because the only thing varying was `file_000123.tmp`.
*   **Flags and signals come from production code.** `build_path_evidence`
    and `apply_existence_evidence` derive `is_system_path` /
    `is_user_path` / `is_application_path` and the signal wording, so the
    path in a row and the flags on it cannot disagree, and `signals` is
    populated exactly as it will be at serving time. Previously the
    generator invented the flags and left `signals` empty in all 10,000
    rows (§78 train/serve skew).
*   **Labels come from the engine.** `make_recommendation` decides; nothing
    here hardcodes an action or a risk (§65).
*   **The corpus contains no evidence production cannot produce.** Two
    consequences, both deliberate:

    - `is_locked` is always `False`, because nothing in production ever
      sets it (ADR 0002, Gap 1). The old corpus carried `is_locked: true`
      on 2,691 rows the model will never see the like of at serving time.
      Locked-file cases live in the red-team set instead, where probing
      beyond the deterministic policy is the point.
    - A file that does not exist has `age_days = None`, because a `stat`
      call on a missing file is what produces that `None`. The old corpus
      gave missing files ages.

The generator has no opinion about policy. It produces evidence; whatever
`policy-v1` says about that evidence is the label. See `docs/policy-v1.md`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from backend.models.system import (
    AICase,
    FileCategory,
    FileRecord,
)
from backend.services.ai_cases import build_ai_case
from backend.services.context import build_decision_context
from backend.services.decision.engine import make_recommendation
from backend.services.evidence import (
    apply_existence_evidence,
    build_path_evidence,
)


@dataclass(frozen=True)
class CaseTemplate:
    """
    One category's worth of plausible file shapes.

    `directories` is chosen so that the path-derived flags vary within a
    category rather than being constant per category: each list mixes
    system paths, user paths, Program Files, and paths matching none of
    them. A model that learned "crash dumps are system files" from a
    corpus where every dump lived under `C:\\Windows` would be learning
    the corpus, not the policy.
    """

    category: FileCategory
    extensions: tuple[str, ...]
    directories: tuple[str, ...]
    stems: tuple[str, ...]


TEMPLATES = (
    CaseTemplate(
        FileCategory.TEMPORARY,
        (".tmp", ".temp", ".part"),
        (
            r"C:\Temp",
            r"C:\Users\Test\AppData\Local\Temp",
            r"C:\Windows\Temp",
            r"D:\Builds\intermediate",
        ),
        ("scratch", "upload", "session", "wct1a2b"),
    ),
    CaseTemplate(
        FileCategory.CACHE,
        (".cache", ".dat", ".idx"),
        (
            r"C:\Users\Test\AppData\Local\Cache",
            r"C:\Users\Test\AppData\Local\Microsoft\Windows\INetCache",
            r"C:\ProgramData\TestApp\Cache",
            r"C:\Windows\SoftwareDistribution\Download",
        ),
        ("thumbnails", "shader", "fontcache", "index"),
    ),
    CaseTemplate(
        FileCategory.LOG,
        (".log", ".etl", ".txt"),
        (
            r"C:\Logs",
            r"C:\Windows\Logs\CBS",
            r"C:\ProgramData\TestApp\Logs",
            r"C:\Users\Test\AppData\Local\TestApp\Logs",
        ),
        ("application", "install", "trace", "service"),
    ),
    CaseTemplate(
        FileCategory.CRASH_DUMP,
        (".dmp", ".mdmp", ".hdmp"),
        (
            r"C:\Dumps",
            r"C:\Windows\Minidump",
            r"C:\Windows\LiveKernelReports",
            r"C:\Users\Test\AppData\Local\CrashDumps",
        ),
        ("memory", "crash", "faulting", "kernel"),
    ),
    CaseTemplate(
        FileCategory.INSTALLER,
        (".msi", ".exe", ".msu"),
        (
            r"C:\Users\Test\Downloads",
            r"C:\Installers",
            r"C:\Windows\Installer",
            r"C:\Program Files\TestApp\redist",
        ),
        ("setup", "vc_redist", "update", "driverpack"),
    ),
    CaseTemplate(
        FileCategory.DRIVER,
        (".sys", ".inf", ".cat"),
        (
            r"C:\Windows\System32\drivers",
            r"C:\Drivers",
            r"C:\Windows\System32\DriverStore\FileRepository",
            r"C:\Program Files\TestVendor\drivers",
        ),
        ("nvlddmkm", "storport", "usbhub", "hdaudio"),
    ),
    CaseTemplate(
        FileCategory.USER_DATA,
        (".pdf", ".docx", ".jpg", ".mp4"),
        (
            r"C:\Users\Test\Documents",
            r"C:\Users\Test\Pictures",
            r"C:\Users\Test\Videos",
            r"D:\Archive\Personal",
        ),
        ("thesis", "invoice", "holiday", "notes"),
    ),
    CaseTemplate(
        FileCategory.APPLICATION_DATA,
        (".db", ".dat", ".bin"),
        (
            r"C:\Program Files\TestApp",
            r"C:\Program Files (x86)\TestApp",
            r"C:\ProgramData\TestApp",
            r"C:\Users\Test\AppData\Roaming\TestApp",
        ),
        ("catalog", "profile", "state", "settings"),
    ),
    CaseTemplate(
        FileCategory.SYSTEM_DATA,
        (".sys", ".etl", ".dat"),
        (
            r"C:\Windows\System32",
            r"C:\Windows\System32\config",
            r"C:\Windows\System32\LogFiles",
            r"D:\SystemBackup",
        ),
        ("pagefile", "registry", "component", "wdi"),
    ),
    CaseTemplate(
        FileCategory.UNKNOWN,
        (".xyz", ".bin", ".000"),
        (
            r"C:\somewhere",
            r"D:\unsorted",
            r"C:\Users\Test\Desktop\misc",
            r"C:\Windows\Temp\unsorted",
        ),
        ("blob", "archive", "export", "chunk"),
    ),
)


# Ages straddle the 30-day policy boundary from both sides at several
# distances, because 30 is the one number at which policy-v1 changes its
# answer. `None` is "exists, but the age could not be read" -- production
# reaches it when `stat` raises.
AGES: tuple[float | None, ...] = (
    None,
    0.0,
    1.0,
    7.0,
    28.0,
    29.0,
    29.9,
    30.0,
    30.1,
    31.0,
    60.0,
    180.0,
    365.0,
    1825.0,
)

# Sizes across seven decades, 4 KB to 15 GB. Policy-v1 ignores size
# entirely, so this exists to keep the model from inventing a size rule:
# "large therefore deletable" is the most obvious wrong heuristic
# available to it, and the corpus has to actively contradict it.
SIZES = (
    4_096,
    65_536,
    1_048_576,
    10_485_760,
    104_857_600,
    1_073_741_824,
    16_106_127_360,
)


@dataclass(frozen=True)
class Combination:
    """
    One point in the evidence space: everything a case is made of except
    its filename stem.

    Two cases built from the same `Combination` ask the model the same
    question, so the generator never emits one twice. That is the whole
    of the "distinct semantic contexts" property: it holds by
    construction rather than by a post-hoc uniqueness check.
    """

    category: FileCategory
    directory: str
    extension: str
    size: int
    exists: bool
    age_days: float | None


def all_combinations() -> list[Combination]:
    """
    Enumerate the whole evidence space in a deterministic order.

    `age_days` is `None` whenever the file does not exist, because that is
    what production produces: `build_file_evidence` cannot stat a file
    that is gone. An age on a missing file would be evidence the model
    never sees at serving time.
    """

    combinations: list[Combination] = []

    for template in TEMPLATES:
        for directory in template.directories:
            for extension in template.extensions:
                for size in SIZES:
                    for age_days in AGES:
                        combinations.append(
                            Combination(
                                category=template.category,
                                directory=directory,
                                extension=extension,
                                size=size,
                                exists=True,
                                age_days=age_days,
                            )
                        )

                    combinations.append(
                        Combination(
                            category=template.category,
                            directory=directory,
                            extension=extension,
                            size=size,
                            exists=False,
                            age_days=None,
                        )
                    )

    return combinations


_STEMS_BY_CATEGORY = {
    template.category: template.stems
    for template in TEMPLATES
}


def build_case(
    combination: Combination,
    stem: str,
) -> AICase:
    """
    Turn one combination into a labelled case using production code only:
    path evidence, then the observation, then the engine, then the case
    builder. Nothing here decides anything.
    """

    path = f"{combination.directory}\\{stem}{combination.extension}"

    file = FileRecord(
        path=path,
        size=combination.size,
        category=combination.category,
        extension=combination.extension,
    )

    evidence = apply_existence_evidence(
        build_path_evidence(file),
        exists=combination.exists,
        age_days=combination.age_days,
    )

    return build_ai_case(
        build_decision_context(
            file,
            evidence,
            make_recommendation(file, evidence),
        )
    )


def generate_cases(
    count: int,
    *,
    seed: int = 42,
) -> list[AICase]:
    """
    Generate `count` semantically distinct labelled cases.

    Raises rather than padding with duplicates when `count` exceeds the
    size of the evidence space.
    """

    corpus, _ = generate_corpus_and_gold(
        count,
        gold_per_category=0,
        seed=seed,
    )

    return corpus


def generate_corpus_and_gold(
    count: int,
    *,
    gold_per_category: int = 10,
    seed: int = 42,
) -> tuple[list[AICase], list[AICase]]:
    """
    Draw the training corpus and a held-out set from one shuffled pool,
    so the two are disjoint by construction rather than by a filter
    somebody could forget to apply.

    The held-out set is stratified: `gold_per_category` cases from each
    `FileCategory`, so it covers evenly even where the corpus does not.

    It is not a *reviewed* gold set. No human has checked these labels;
    they are the engine's, same as the corpus. See `docs/dataset-card.md`
    -- calling it human-reviewed before anyone reviews it would be a
    false claim (§555).
    """

    if count < 1:
        raise ValueError("count must be at least 1")

    if gold_per_category < 0:
        raise ValueError("gold_per_category must not be negative")

    rng = random.Random(seed)

    pool = all_combinations()
    rng.shuffle(pool)

    needed = {category: gold_per_category for category in FileCategory}

    gold: list[Combination] = []
    remaining: list[Combination] = []

    for combination in pool:
        if needed[combination.category] > 0:
            needed[combination.category] -= 1
            gold.append(combination)
            continue

        remaining.append(combination)

    short = {
        category.value: missing
        for category, missing in needed.items()
        if missing > 0
    }

    if short:
        raise ValueError(
            "the evidence space cannot supply the held-out set: "
            f"{short}"
        )

    if count > len(remaining):
        raise ValueError(
            f"count={count} exceeds the {len(remaining)} distinct "
            "evidence combinations available after the held-out set. "
            "Widen TEMPLATES, AGES, or SIZES -- do not repeat cases."
        )

    return (
        [
            build_case(combination, _stem_for(combination, rng))
            for combination in remaining[:count]
        ],
        [
            build_case(combination, _stem_for(combination, rng))
            for combination in gold
        ],
    )


def _stem_for(
    combination: Combination,
    rng: random.Random,
) -> str:
    return rng.choice(_STEMS_BY_CATEGORY[combination.category])
