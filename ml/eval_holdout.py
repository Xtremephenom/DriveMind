"""
Run a Hugging Face model over DriveMind's held-out sets.

    python -m ml.eval_holdout --model Qwen/Qwen3-4B
    python -m ml.eval_holdout --model Qwen/Qwen3-4B --adapter artifacts/qlora
    python -m ml.eval_holdout --smoke          # 12 cases, not a measurement

The measurement itself lives in `backend.services.ai_holdout`. This file
is argument parsing, a model, and a stopwatch -- deliberately, because a
CLI that reimplements any part of the scoring is a second scorer that can
disagree with the first (§272/§528). Everything reported here is produced
by `format_held_out_report` from evaluations the same evaluator computes
for `RuleBasedAIProvider`, so a base-model run, a fine-tuned run, and the
deterministic reference are all directly comparable.

There is no flag to shorten the held-out sets. `evaluate_emitter` proves
the cases it scores are the published rows before it scores them, and a
`--limit 20` would either break that proof or bypass it -- and then print
a report headed "held-out evaluation". `--smoke` exists for the "does the
model load and emit anything" question and routes to the 12 hand-built
baseline cases, under a banner that says what they are.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from backend.services.ai_baseline import format_report, run_baseline
from backend.services.ai_holdout import (
    HELD_OUT,
    HoldoutMismatch,
    evaluate_emitter,
    format_held_out_report,
)
from ml.hf_runner import (
    DEFAULT_MODEL_ID,
    GenerationConfig,
    ModelBundle,
    load_model,
    make_emitter,
)

# 1,000 test + 100 gold + 26 red_team. Hardcoded only for the pre-run
# estimate; the real counts come from the verification step.
HELD_OUT_CASES = 1_126


def describe_environment() -> list[str]:
    """
    What the run happened on.

    A held-out figure is only comparable with another if the runtime is
    known (§514/§516) -- 4-bit on one card and bf16 on another are
    different measurements of the same weights. Reported as lines rather
    than logged, so it lands in the same output as the numbers.
    """

    lines = [f"  python          {sys.version.split()[0]}"]

    try:
        import torch

    except ImportError:
        lines.append("  torch           not installed")
        return lines

    lines.append(f"  torch           {torch.__version__}")

    if not torch.cuda.is_available():
        lines.append("  device          cpu (no CUDA device visible)")
        return lines

    # The physical card, not the index this process sees. Two runs pinned
    # to different cards of the same machine both report "device 0",
    # because CUDA renumbers what is visible from zero -- so the mask is
    # the only thing in the report that distinguishes them. It matters as
    # soon as a base run and a tuned run are placed on separate cards
    # (§514/§516).
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible is not None:
        lines.append(f"  visible cards   CUDA_VISIBLE_DEVICES={visible!r}")

    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)

        lines.append(
            f"  device {index}        {properties.name}, "
            f"{properties.total_memory // (1024 * 1024)} MiB, "
            f"CUDA {torch.version.cuda}"
        )

    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.eval_holdout",
        description=(
            "Evaluate a Hugging Face causal LM on DriveMind's held-out "
            "sets, verifying first that the cases scored are the "
            "published ones."
        ),
    )

    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a LoRA adapter to load on top of --model",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help=(
            "load unquantized instead of 4-bit NF4 -- bf16 where the GPU "
            "supports it, fp16 on Turing. Needs roughly 9 GB for a 4B "
            "model, so not the laptop path."
        ),
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "enable the chat template's thinking mode. Expect this to "
            "destroy structured-output validity: reasoning prose is not "
            "the JSON contract, and nothing here strips it out."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="enables sampling; omit for greedy, reproducible decoding",
    )
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the 12 hand-built baseline cases instead. A load test, "
            "not a measurement of the model."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report to this path",
    )

    return parser


def bundle_from_args(args: argparse.Namespace) -> ModelBundle:
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature is not None,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    return load_model(
        args.model,
        load_in_4bit=not args.no_4bit,
        adapter_path=args.adapter,
        enable_thinking=args.thinking,
        generation=generation,
    )


def run_smoke(bundle: ModelBundle) -> str:
    """
    The 12 baseline cases, through the existing baseline runner.

    Routed to `run_baseline` rather than to a shortened held-out set, and
    the banner is not decoration: these 12 cases were hand-built to cover
    the obvious shapes, and a model can pass all of them by pattern
    matching on `category`. It answers "does the model load, and does it
    emit anything resembling the contract".
    """

    evaluations, summary = run_baseline(make_emitter(bundle))

    return "\n".join(
        [
            "SMOKE RUN -- 12 hand-built baseline cases.",
            "This is not the held-out measurement. It cannot distinguish",
            "a competent model from a lucky one, and no figure from it",
            "belongs in docs/model-card.md.",
            "",
            f"provider: {bundle.describe()}",
            "",
            format_report(evaluations, summary),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("DriveMind model evaluation")
    print("\n".join(describe_environment()))
    print()

    if args.temperature is not None:
        print(
            "  NOTE  sampling is on, so this run is not reproducible.\n"
            "        Omit --temperature for the figure you report.\n"
        )

    started = time.perf_counter()
    bundle = bundle_from_args(args)
    loaded = time.perf_counter()

    print(f"  loaded in {loaded - started:.1f}s: {bundle.describe()}")

    if args.smoke:
        report = run_smoke(bundle)

    else:
        print(
            f"  generating {HELD_OUT_CASES} responses over "
            f"{len(HELD_OUT)} held-out sets, one at a time"
        )

        try:
            results, verified = evaluate_emitter(make_emitter(bundle))

        except HoldoutMismatch as exc:
            # Not caught to be tidy. This is the failure that says the
            # rows on disk are not the cases just regenerated, and the
            # only correct response is to refuse to print a number.
            print(f"\nHELD-OUT VERIFICATION FAILED\n\n{exc}")
            return 2

        report = format_held_out_report(
            results,
            bundle.describe(),
            verified,
        )

    generation_seconds = time.perf_counter() - loaded

    print()
    print(report)

    cases = 12 if args.smoke else HELD_OUT_CASES

    timing = (
        f"  generation      {generation_seconds / 60:.1f} min for {cases} "
        f"cases ({generation_seconds / cases:.2f}s each)"
    )

    print()
    print(timing)

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
                    timing,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        print(f"  written to      {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

