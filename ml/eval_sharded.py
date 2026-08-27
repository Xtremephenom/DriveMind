"""
Run one held-out evaluation across every available GPU.

    python -m ml.eval_sharded --model Qwen/Qwen3-4B --devices 0,1 \\
        --out artifacts/base_holdout.txt

    python -m ml.eval_sharded --self-test      # CPU, no model, ~1 min

`ml.eval_holdout` generates 1,126 responses one at a time on one card --
the throughput ceiling `hf_runner.make_emitter` names in its own
docstring. On a 2x T4 box that leaves half the machine idle for the
better part of an hour. This module splits the *generation* across cards
and keeps the *scoring* in one place.

Splitting a measurement is how measurements get quietly corrupted, so
the shape of the code is the argument that this one is not:

*   **Generation and scoring are separate passes.** A worker generates
    raw text and writes it down. No worker scores anything -- each one
    computes a summary over its own shard and throws it away, because
    letting the evaluator run is cheaper than reimplementing its case
    enumeration.
*   **Scoring is one `evaluate_emitter` call, over the whole held-out
    set, in one process.** It regenerates all 1,126 cases, verifies every
    one against the row on disk, and scores every one -- the same call an
    unsharded run makes. There is no `--limit`, and the scorer is not
    told that shards exist (§272/§528).
*   **Both passes enumerate cases by calling `evaluate_emitter`.** Not by
    a second copy of its iteration order. A worker decides "mine or not"
    from its call index, so the sharding rides on the evaluator's own
    enumeration and cannot drift from it.
*   **Any hole is a refusal.** A case no worker recorded, a case two
    workers both recorded, a shard that ran a different model: each one
    raises, and no report is printed. Scoring an ungenerated case would
    count an empty string as a failed response and report it as the
    model's (§404).

`--self-test` checks that on CPU with no model and no GPU: it runs the
deterministic rule-based provider through the two-pass sharded path and
through a plain single-process `evaluate_emitter`, and requires the two
reports to be byte-identical. That is the property this file exists to
preserve, and it is the one thing about it that can be verified on a
laptop.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from backend.models.system import AICase
from backend.services.ai_holdout import (
    HoldoutMismatch,
    SplitResult,
    evaluate_emitter,
    format_held_out_report,
)
from ml.eval_holdout import describe_environment
from ml.hf_runner import DEFAULT_MODEL_ID

# 1,000 test + 100 gold + 26 red_team, for the pre-run estimate only. The
# reported counts come from the verification step, same as everywhere else.
HELD_OUT_CASES = 1_126


class ShardError(RuntimeError):
    """The shards do not cover the held-out set. Nothing has been scored."""


def generate_shard(
    emit_text: Callable[[AICase], str],
    *,
    shard: int,
    shards: int,
    progress_every: int = 25,
) -> list[dict]:
    """
    Generate this shard's share of the held-out cases. Score nothing.

    The evaluator drives the loop and this function decides, per call,
    whether the case belongs to this shard. That inversion is the point:
    the alternative is a second function that walks `HELD_OUT` and each
    split in the same order `run_held_out` does, which is a copy of the
    evaluator's enumeration that agrees with it until someone edits one
    of them (§272/§528).

    `index % shards` interleaves rather than splitting into contiguous
    blocks. Prompt length and case difficulty are not uniform across a
    split -- `red_team` sits entirely at the end of the enumeration -- so
    contiguous blocks would hand one card the adversarial tail and leave
    the other waiting for it.

    Cases belonging to another shard get the empty string. The evaluator
    duly scores it and the score is discarded with the rest of this
    worker's summary; only `recorded` leaves this function.
    """

    recorded: list[dict] = []
    calls = itertools.count()
    started = time.perf_counter()

    def emit(case: AICase) -> str:
        index = next(calls)

        if index % shards != shard:
            return ""

        text = emit_text(case)

        recorded.append(
            {
                "index": index,
                "case_id": case.case_id,
                "response": text,
            }
        )

        if len(recorded) % progress_every == 0:
            elapsed = time.perf_counter() - started

            print(
                f"  shard {shard + 1}/{shards}  {len(recorded)} generated,"
                f" {elapsed / len(recorded):.2f}s each,"
                f" {elapsed / 60:.1f} min in",
                flush=True,
            )

        return text

    # Verification is not discarded -- `evaluate_emitter` proves the rows
    # on disk are the cases it is about to enumerate, and a worker that
    # skipped that would generate against a corpus the scorer will reject.
    evaluate_emitter(emit)

    return recorded


def merge_shards(shards: list[dict]) -> tuple[dict[str, str], str]:
    """
    One `case_id -> response` map from every shard, or a refusal.

    Returns the provider description too, and checks that every shard
    reports the same one. That check is not bookkeeping: the parent passes
    `--adapter` to each worker separately, and a typo in one of them
    produces a run where half the cases were answered by the base model
    and half by the fine-tune. The numbers would look plausible and mean
    nothing.

    `case_id` is unique across all three held-out splits -- the corpus
    build refuses to write on a duplicate or a cross-file overlap -- so it
    is a safe key, and keying by it rather than by call index makes the
    merge independent of the order the shards happen to be read in.
    """

    if not shards:
        raise ShardError("no shards to merge")

    # The expected count comes from what the workers *declared*, not from
    # how many files were handed to this function. Trusting `len(shards)`
    # would make one file of a two-shard run look like a complete
    # one-shard run: the indices would be `[0]`, `range(1)` would accept
    # them, and the missing half would only surface downstream as 563
    # ungenerated cases. Same failure, named two steps later and worse.
    declared = {shard["shards"] for shard in shards}

    if len(declared) != 1:
        raise ShardError(
            "the shards disagree about how many shards there were: "
            f"{sorted(declared)}. These files are from different runs."
        )

    expected = declared.pop()
    seen = sorted(shard["shard"] for shard in shards)

    if seen != list(range(expected)):
        raise ShardError(
            f"expected shards {list(range(expected))}, got {seen}. A shard "
            "is missing or duplicated, so part of the held-out set was "
            "never generated."
        )

    providers = {shard["provider"] for shard in shards}

    if len(providers) != 1:
        raise ShardError(
            "the shards did not run the same model: "
            + " | ".join(sorted(providers))
            + ". Every case in one report has to come from one "
            "configuration or the report describes nothing."
        )

    cache: dict[str, str] = {}

    for shard in shards:
        for record in shard["records"]:
            case_id = record["case_id"]

            if case_id in cache:
                raise ShardError(
                    f"{case_id} was generated by more than one shard, so "
                    "the shards overlap -- which means at least one other "
                    "case was generated by none of them."
                )

            cache[case_id] = record["response"]

    if not cache:
        raise ShardError("the shards are empty; nothing was generated")

    return cache, providers.pop()


def score_from_cache(
    cache: dict[str, str],
) -> tuple[list[SplitResult], dict[str, int]]:
    """
    The authoritative pass. Every case, verified, scored once, here.

    This is a plain `evaluate_emitter` call whose emitter is a dictionary
    lookup. The evaluator cannot tell it apart from a model, which is the
    reason the report it produces is comparable with an unsharded one --
    and the reason there is no shard-aware scoring code anywhere.

    Two ways the cache can fail to line up with the held-out set, and
    both raise:

    *   a case nobody generated -- returning `""` for it would have the
        evaluator record a failed parse and attribute it to the model
    *   a generated response no case asked for -- the shards and the
        evaluator disagree about which cases exist, and any report from
        that state is over a set nobody can name

    Misses are collected rather than raised on first sight, so the error
    says how bad the hole is instead of naming one case and leaving the
    operator to re-run for the next.
    """

    missing: list[str] = []
    used: set[str] = set()

    def emit(case: AICase) -> str:
        if case.case_id not in cache:
            missing.append(case.case_id)

            return ""

        used.add(case.case_id)

        return cache[case.case_id]

    results, verified = evaluate_emitter(emit)

    if missing:
        raise ShardError(
            f"{len(missing)} of {len(missing) + len(used)} held-out cases "
            f"were never generated (first: {missing[:5]}). No report is "
            "printed: an empty response scores as a failed parse, and "
            "reporting that as the model's would understate it for a "
            "reason that has nothing to do with the model."
        )

    unused = sorted(set(cache) - used)

    if unused:
        raise ShardError(
            f"{len(unused)} generated responses match no held-out case "
            f"(first: {unused[:5]}). The shards and the evaluator disagree "
            "about which cases exist."
        )

    return results, verified


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.eval_sharded",
        description=(
            "Evaluate a Hugging Face causal LM on DriveMind's held-out "
            "sets, generating across every GPU and scoring in one place."
        ),
    )

    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)

    # No --temperature and no --top-p. Sampling is not the figure anyone
    # reports, and `ml.eval_holdout` is where a temperature sweep belongs;
    # keeping this file to the reproducible path means one less way for a
    # sharded run and an unsharded run to differ.
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "comma-separated CUDA device indices, one shard each "
            "(default: every visible card). A single device is allowed "
            "and works; it is just `ml.eval_holdout` with an extra corpus "
            "regeneration."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "where the per-shard generations are written (default: a "
            "directory beside --out, or ./artifacts/_shards). Kept rather "
            "than deleted: they are the raw model output behind the "
            "report, and re-earning them costs the whole run."
        ),
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "run the sharded path and the single-process path with the "
            "deterministic rule-based provider and require byte-identical "
            "reports. No GPU, no model, no network."
        ),
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def run_worker(args: argparse.Namespace) -> int:
    """
    One shard, one card, one file of raw generations.

    Loads the model itself rather than being handed one, because it is a
    separate process -- which is the only way two cards run at once
    without CUDA context juggling, and it means an OOM on one card kills
    one worker with a stack trace instead of the notebook kernel.
    """

    from ml.eval_holdout import bundle_from_args
    from ml.hf_runner import make_emitter

    # `bundle_from_args` reads these two; this file does not offer them.
    args.temperature = None
    args.top_p = None

    started = time.perf_counter()
    bundle = bundle_from_args(args)
    loaded = time.perf_counter()

    print(
        f"  shard {args.shard + 1}/{args.shards} loaded in "
        f"{loaded - started:.1f}s: {bundle.describe()}",
        flush=True,
    )

    records = generate_shard(
        make_emitter(bundle),
        shard=args.shard,
        shards=args.shards,
    )

    seconds = time.perf_counter() - loaded

    args.cache.parent.mkdir(parents=True, exist_ok=True)

    args.cache.write_text(
        json.dumps(
            {
                "shard": args.shard,
                "shards": args.shards,
                "provider": bundle.describe(),
                "visible_cards": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cases": len(records),
                "seconds": seconds,
                "records": records,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"  shard {args.shard + 1}/{args.shards} done: {len(records)} cases "
        f"in {seconds / 60:.1f} min ({seconds / max(len(records), 1):.2f}s "
        f"each) -> {args.cache}",
        flush=True,
    )

    return 0


def resolve_devices(spec: str | None) -> list[str]:
    """Which cards to shard over, as CUDA_VISIBLE_DEVICES values."""

    if spec:
        devices = [part.strip() for part in spec.split(",") if part.strip()]

        if not devices:
            raise ShardError(f"--devices {spec!r} names no device")

        return devices

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise ShardError(
            "no CUDA device is visible, so there is nothing to shard over. "
            "Run `python -m ml.eval_holdout` for the single-process path, "
            "or `--self-test` to check this file without a GPU."
        )

    return [str(index) for index in range(torch.cuda.device_count())]


@dataclass
class Worker:
    shard: int
    device: str
    result_path: Path
    log_path: Path
    process: subprocess.Popen
    stream: IO[str]


def last_line(path: Path) -> str:
    """The most recent non-empty line of a log a child is still writing."""

    try:
        lines = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            if line.strip()
        ]

    except OSError:
        return "(log unreadable)"

    return lines[-1] if lines else "(no output yet)"


def spawn_workers(
    args: argparse.Namespace,
    devices: list[str],
    cache_dir: Path,
) -> list[Worker]:
    """One subprocess per card, each pinned by CUDA_VISIBLE_DEVICES."""

    cache_dir.mkdir(parents=True, exist_ok=True)

    workers: list[Worker] = []

    for shard, device in enumerate(devices):
        command = [
            sys.executable,
            "-m",
            "ml.eval_sharded",
            "--worker",
            "--shard",
            str(shard),
            "--shards",
            str(len(devices)),
            "--cache",
            str(cache_dir / f"shard_{shard}.json"),
            "--model",
            args.model,
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]

        if args.adapter:
            command += ["--adapter", str(args.adapter)]

        if args.no_4bit:
            command.append("--no-4bit")

        if args.thinking:
            command.append("--thinking")

        log_path = cache_dir / f"shard_{shard}.log"
        stream = log_path.open("w", encoding="utf-8")

        workers.append(
            Worker(
                shard=shard,
                device=device,
                result_path=cache_dir / f"shard_{shard}.json",
                log_path=log_path,
                process=subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        # The one line that makes this parallel. Each
                        # worker sees exactly one card and calls it cuda:0,
                        # so `device_map="auto"` inside `load_model` has
                        # only one place to put the weights and no reason
                        # to shard a replica.
                        "CUDA_VISIBLE_DEVICES": device,
                        "TOKENIZERS_PARALLELISM": "false",
                    },
                ),
                stream=stream,
            )
        )

        print(f"  shard {shard + 1}/{len(devices)} -> card {device}")

    return workers


def wait_for_workers(workers: list[Worker], *, heartbeat: float = 30.0) -> None:
    """
    Block until every worker exits, reporting progress, then refuse if any
    of them failed.

    The heartbeat exists because the alternative on a notebook is a cell
    that prints nothing for forty minutes, which is indistinguishable from
    a hang -- and a hung 2-card run on a session-limited runtime is worth
    noticing early. Each worker's own progress line is echoed rather than
    reformatted, so what appears here is what is in its log.
    """

    started = time.perf_counter()

    while any(worker.process.poll() is None for worker in workers):
        time.sleep(heartbeat)

        print(f"\n  [{(time.perf_counter() - started) / 60:.1f} min]")

        for worker in workers:
            state = (
                "running"
                if worker.process.poll() is None
                else f"exit {worker.process.returncode}"
            )

            print(
                f"    card {worker.device} ({state}): "
                f"{last_line(worker.log_path)}"
            )

    for worker in workers:
        worker.stream.close()

    failed = [
        worker for worker in workers if worker.process.returncode != 0
    ]

    if not failed:
        return

    detail = "\n\n".join(
        f"card {worker.device} exited {worker.process.returncode}; tail of "
        f"{worker.log_path}:\n"
        + "\n".join(
            worker.log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()[-20:]
        )
        for worker in failed
    )

    raise ShardError(
        f"{len(failed)} of {len(workers)} shards failed, so part of the "
        f"held-out set was never generated.\n\n{detail}"
    )


def run_parent(args: argparse.Namespace) -> int:
    devices = resolve_devices(args.devices)

    cache_dir = args.cache_dir or (
        args.out.parent / f"_shards_{args.out.stem}"
        if args.out
        else Path("artifacts") / "_shards"
    )

    print("DriveMind model evaluation -- sharded generation")
    print("\n".join(describe_environment()))
    print()
    print(
        f"  generating {HELD_OUT_CASES} responses across {len(devices)} "
        f"card(s), about {HELD_OUT_CASES // len(devices)} each"
    )

    workers = spawn_workers(args, devices, cache_dir)

    started = time.perf_counter()
    wait_for_workers(workers)
    generation_seconds = time.perf_counter() - started

    shards = [
        json.loads(worker.result_path.read_text(encoding="utf-8"))
        for worker in workers
    ]

    cache, provider = merge_shards(shards)

    print(
        f"\n  merged {len(cache)} generations from {len(shards)} shard(s); "
        "scoring all of them in one process"
    )

    results, verified = score_from_cache(cache)
    report = format_held_out_report(results, provider, verified)

    card_seconds = sum(shard["seconds"] for shard in shards)

    placement = [
        "  generation placement:",
        *(
            f"    card {shard['visible_cards']}  {shard['cases']:>4} cases"
            f"  {shard['seconds'] / 60:>5.1f} min"
            f"  {shard['seconds'] / max(shard['cases'], 1):.2f}s each"
            for shard in sorted(shards, key=lambda entry: entry["shard"])
        ),
        f"  scoring         one process, all {len(cache)} cases, one "
        "evaluate_emitter call",
    ]

    timing = [
        f"  generation      {generation_seconds / 60:.1f} min wall clock "
        f"for {len(cache)} cases across {len(devices)} card(s)",
        f"                  {card_seconds / 60:.1f} card-minutes, "
        f"{card_seconds / len(cache):.2f}s per case per card",
    ]

    print()
    print(report)
    print()
    print("\n".join(placement))
    print("\n".join(timing))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)

        args.out.write_text(
            "\n".join(
                [
                    "DriveMind model evaluation",
                    *describe_environment(),
                    "",
                    report,
                    "",
                    *placement,
                    *timing,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        print(f"  written to      {args.out}")

    return 0


SELF_TEST_LABEL = "self-test: RuleBasedAIProvider"


def as_shard(records: list[dict], shard: int, shards: int) -> dict:
    """A worker's output file, without a worker. Used by `--self-test`."""

    return {
        "shard": shard,
        "shards": shards,
        "provider": SELF_TEST_LABEL,
        "visible_cards": None,
        "cases": len(records),
        "seconds": 0.0,
        "records": records,
    }


def expect_refusal(what: str, call: Callable[[], object]) -> bool:
    """Require `call` to raise `ShardError`. Print which way it went."""

    try:
        call()

    except ShardError:
        print(f"    refused {what}")

        return True

    print(f"    DID NOT REFUSE {what} -- this file's guarantee is not real")

    return False


def self_test() -> int:
    """
    Prove the sharded path scores identically to the single-process one.

    Run with the rule-based provider, which is a deterministic mirror of
    the engine, so every figure is 1.0 or 0.0 and none of them is
    interesting. What is being checked is not the numbers but that
    splitting generation across N shards and scoring the merge produces
    the *same report text* as not splitting it at all -- for N of 1, 2, 3
    and 4, none of which divides 1,126 evenly.

    Then the refusals, exercised rather than asserted (§589): a hole, an
    overlap, a stray response, two shards that ran different models, a
    missing shard, and two files from different runs. A guard nobody has
    watched fail is indistinguishable from one that cannot fire.
    """

    from backend.services.ai_baseline import provider_as_text
    from backend.services.ai_rule_based import RuleBasedAIProvider

    emit = provider_as_text(RuleBasedAIProvider())

    print("DriveMind sharded evaluation self-test")
    print("  no GPU, no model, no network -- the rule-based provider")
    print()

    reference_results, reference_verified = evaluate_emitter(emit)

    reference = format_held_out_report(
        reference_results,
        SELF_TEST_LABEL,
        reference_verified,
    )

    print(
        f"  reference       single process, "
        f"{sum(reference_verified.values())} cases"
    )

    ok = True
    baseline_shards: list[dict] = []

    for shards in (1, 2, 3, 4):
        collected = [
            as_shard(
                generate_shard(
                    emit,
                    shard=shard,
                    shards=shards,
                    progress_every=10**9,
                ),
                shard,
                shards,
            )
            for shard in range(shards)
        ]

        cache, provider = merge_shards(collected)
        results, verified = score_from_cache(cache)
        report = format_held_out_report(results, provider, verified)

        sizes = "+".join(str(entry["cases"]) for entry in collected)

        if report == reference:
            print(f"  {shards} shard(s)      identical  ({sizes} cases)")

        else:
            ok = False
            print(f"  {shards} shard(s)      DIFFERS  ({sizes} cases)")

            for line_number, (left, right) in enumerate(
                zip(reference.splitlines(), report.splitlines()),
                start=1,
            ):
                if left != right:
                    print(f"    first difference at line {line_number}:")
                    print(f"      single  {left!r}")
                    print(f"      sharded {right!r}")

                    break

        if shards == 2:
            baseline_shards = collected

    print("  refusals:")

    holed = [dict(entry) for entry in baseline_shards]
    holed[0] = {**holed[0], "records": holed[0]["records"][1:]}
    ok &= expect_refusal(
        "a case no shard generated",
        lambda: score_from_cache(merge_shards(holed)[0]),
    )

    overlapped = [dict(entry) for entry in baseline_shards]
    overlapped[1] = {
        **overlapped[1],
        "records": (
            overlapped[1]["records"] + [baseline_shards[0]["records"][0]]
        ),
    }
    ok &= expect_refusal(
        "the same case in two shards",
        lambda: merge_shards(overlapped),
    )

    stray = [dict(entry) for entry in baseline_shards]
    stray[0] = {
        **stray[0],
        "records": (
            stray[0]["records"]
            + [{"index": -1, "case_id": "not-a-case", "response": "{}"}]
        ),
    }
    ok &= expect_refusal(
        "a response matching no held-out case",
        lambda: score_from_cache(merge_shards(stray)[0]),
    )

    mixed = [dict(entry) for entry in baseline_shards]
    mixed[1] = {**mixed[1], "provider": SELF_TEST_LABEL + "  lora=elsewhere"}
    ok &= expect_refusal(
        "shards that ran different models",
        lambda: merge_shards(mixed),
    )

    missing_shard = [dict(baseline_shards[0])]
    ok &= expect_refusal(
        "a missing shard",
        lambda: merge_shards(missing_shard),
    )

    from_two_runs = [
        dict(baseline_shards[0]),
        {**baseline_shards[1], "shards": 3},
    ]
    ok &= expect_refusal(
        "files from two different runs",
        lambda: merge_shards(from_two_runs),
    )

    print()
    print("  PASS" if ok else "  FAIL")

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.self_test:
            return self_test()

        if args.worker:
            return run_worker(args)

        return run_parent(args)

    except HoldoutMismatch as exc:
        # The rows on disk are not the cases just regenerated. The only
        # correct response is to refuse to print a number.
        print(f"\nHELD-OUT VERIFICATION FAILED\n\n{exc}")

        return 2

    except ShardError as exc:
        print(f"\nSHARDED EVALUATION REFUSED\n\n{exc}")

        return 3


if __name__ == "__main__":
    raise SystemExit(main())
