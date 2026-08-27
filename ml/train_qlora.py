"""
QLoRA fine-tune of a causal LM on DriveMind's training split.

    python -m ml.train_qlora --model Qwen/Qwen3-4B --out artifacts/qlora

    python -m torch.distributed.run --nproc_per_node=2 \\
        -m ml.train_qlora --model Qwen/Qwen3-4B --out artifacts/qlora

    python -m ml.eval_holdout --model Qwen/Qwen3-4B --adapter artifacts/qlora

Written for 2x Tesla T4 (Kaggle), which is the constraint that shapes
every default here: 16 GB per card, no bfloat16, no flash-attention. The
development machine is a 4 GB RTX 3050 -- enough to run inference with
offload, not enough to train a 4B model, which is why this script is
written to be *read* locally and *run* elsewhere.

The second form above is **data** parallelism -- one full model replica
per card, gradients all-reduced, roughly half the wall clock. It is not
the same thing as `device_map="auto"` across two cards, which is *model*
parallelism: that splits one replica's layers over both devices and then
runs them in sequence, so card 1 waits while card 0 computes and the pair
is slower than a single card once transfers are counted. A 4B model in
4-bit NF4 is ~2.5 GB and fits one 16 GB T4 many times over, so there is
nothing to split and everything to replicate.

`--grad-accum` is per rank, so the effective batch is
`batch_size * grad_accum * world_size`. Halve `--grad-accum` when you
double the process count or you have silently doubled the batch and
changed the recipe. `drivemind_run.json` records `world_size` alongside
the effective batch for exactly this reason: a run at world size 2 does
not produce the same adapter as the same flags at world size 1, because
the data order and the gradient reduction both differ (§514/§516).

Four choices that are easy to get wrong, and wrong in ways that produce a
plausible-looking number:

*   **Loss is masked to the completion.** Labels over the prompt tokens
    are `-100`. Every DriveMind prompt shares a ~40-line system
    instruction, so unmasked training spends most of its gradient
    teaching the model to recite text it is always given for free, and
    the reported training loss becomes mostly a measure of how well it
    memorised the boilerplate.
*   **The prompt is rendered by `hf_runner.render_prompt`.** The same
    function the evaluator calls. Not a second formatter that agrees with
    it today (§78).
*   **Checkpoint selection uses `validation.jsonl`, never
    `test.jsonl`.** Selection is a form of fitting; a checkpoint chosen
    by test-set performance has consumed the test set, and the held-out
    number afterwards is no longer held out.
*   **`train.jsonl` is the only file read for training.** The evaluator
    cannot even be pointed at it -- `ai_holdout.HELD_OUT` omits train and
    validation deliberately -- and this script is the other half of that
    boundary.

Selection is on `eval_loss`, which is a proxy: lower loss on validation
is not the same as higher action agreement. It is used because measuring
agreement needs generation, and generating over 1,000 validation cases at
every checkpoint costs more than the whole fine-tune. The metrics that
matter are measured once, afterwards, by `ml.eval_holdout` -- and those
are the only numbers that belong in a model card.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.decision.engine import POLICY_VERSION
from ml.hf_runner import (
    DEFAULT_MODEL_ID,
    pick_dtype,
    render_prompt,
    render_target,
)

DATA_DIR = Path("data")

# Long enough for the longest rendered prompt plus a contract response.
# Measured rather than guessed -- `report_lengths` prints the distribution
# and refuses to train if anything would be truncated, because a
# truncated example teaches the model to stop mid-JSON.
DEFAULT_MAX_LENGTH = 1_024

# Every linear projection in the block, attention and MLP both. LoRA on
# attention alone is the older recipe and consistently underperforms on
# format-following tasks, which is exactly what this is.
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TrainingDataError(RuntimeError):
    """The rows on disk are not usable for training. Nothing has run."""


def ddp_world() -> tuple[int, int, int]:
    """
    `(local_rank, rank, world_size)` as the launcher set them.

    Read from the environment rather than from `torch.distributed`
    because `device_map` has to be decided *before* the process group
    exists -- `Trainer` is what initialises it, and by then the weights
    are already placed. `torchrun` sets all three variables; a plain
    `python -m` run sets none, and the fallbacks make that path
    byte-identical to what this script did before it could run
    distributed at all.
    """

    return (
        int(os.environ.get("LOCAL_RANK", 0)),
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
    )


def announce(*parts: object) -> None:
    """
    Print on rank 0 only.

    Two ranks writing the same line to one terminal produces output where
    every line appears twice and neither copy says which process wrote
    it. Worse on a notebook, where the two streams interleave mid-line.
    The rank is re-read each call rather than captured, so this is safe
    to use from module-level functions that run before `main` has decided
    anything.
    """

    if int(os.environ.get("RANK", 0)) == 0:
        print(*parts)


def load_rows(path: Path) -> list[dict]:
    """
    Read one JSONL split, checking the policy stamp on every row.

    The stamp is checked here and not only in `ai_holdout` because a
    training run is where a stale corpus does the most damage: it
    finishes, produces an adapter, and the adapter has learned the
    previous policy's labels. Nothing downstream would say so -- the
    fine-tuned model would simply disagree with the engine and look like
    a weak model rather than a mislabelled one (§516).
    """

    if not path.exists():
        raise TrainingDataError(
            f"{path} is missing. `data/` is gitignored, so a clone has no "
            "rows until the corpus is rebuilt:\n"
            "  python -c \"from backend.services.dataset.build import "
            'build_dataset; build_dataset()"'
        )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not rows:
        raise TrainingDataError(f"{path} is empty")

    stale = {
        row["policy_version"]
        for row in rows
        if row["policy_version"] != POLICY_VERSION
    }

    if stale:
        raise TrainingDataError(
            f"{path} carries policy version(s) {sorted(stale)}, engine is "
            f"{POLICY_VERSION}. Rebuild the corpus; training on labels "
            "from another policy produces an adapter that disagrees with "
            "the engine by construction."
        )

    return rows


@dataclass
class Example:
    """One tokenized row: the whole sequence, and what to score."""

    input_ids: list[int]
    labels: list[int]


def encode_row(
    tokenizer: Any,
    row: dict,
    *,
    enable_thinking: bool = False,
) -> Example:
    """
    Tokenize one row into a sequence with the prompt masked out.

    `add_special_tokens=False` on both halves is not an optimisation. The
    chat template has already inserted every framing token the model
    expects; letting the tokenizer add its own on top produces a
    duplicated BOS and a training distribution the model never sees at
    inference. It is a silent bug -- loss still falls, generation is just
    subtly worse.

    The target ends with `eos_token_id`, appended as an id rather than as
    text. For a Qwen chat model that token is `<|im_end|>`: the same token
    `apply_chat_template` uses to close an assistant turn. Training
    without it produces a model that emits correct JSON and then keeps
    going, and "keeps going" fails `parse_ai_response` as hard as
    malformed JSON does. Appending the id sidesteps the question of
    whether a tokenizer recognises a special token spelled out inside a
    string, which is behaviour that varies.
    """

    prompt_text = render_prompt(
        tokenizer,
        row["prompt"],
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    target_ids = tokenizer(
        render_target(row["response"]),
        add_special_tokens=False,
    )["input_ids"] + [tokenizer.eos_token_id]

    return Example(
        input_ids=prompt_ids + target_ids,
        labels=[-100] * len(prompt_ids) + list(target_ids),
    )


def encode_split(
    tokenizer: Any,
    rows: list[dict],
    *,
    name: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    enable_thinking: bool = False,
) -> list[Example]:
    """
    Tokenize a split, and **refuse** rather than truncate.

    `max_length=1024` with silent truncation would drop the tail of the
    longest prompts -- which is where `signals` lives, the part of the
    evidence the label most depends on. The model would then be trained
    to answer a question it was not shown. Raising means the operator
    either raises the budget or finds out the corpus changed shape.
    """

    examples = [
        encode_row(tokenizer, row, enable_thinking=enable_thinking)
        for row in rows
    ]

    lengths = [len(example.input_ids) for example in examples]
    over = [length for length in lengths if length > max_length]

    if over:
        raise TrainingDataError(
            f"{name}: {len(over)} of {len(examples)} rows exceed "
            f"max_length={max_length} (longest {max(lengths)} tokens). "
            "Raise --max-length; truncating would cut evidence the label "
            "depends on."
        )

    scored = sum(1 for label in examples[0].labels if label != -100)

    announce(
        f"  {name:<11} {len(examples)} rows, tokens min {min(lengths)}"
        f" / mean {sum(lengths) // len(lengths)} / max {max(lengths)},"
        f" {scored} scored on row 0"
    )

    return examples


class Collator:
    """
    Pad a batch to its longest member.

    Written out rather than reached for from `transformers`, because the
    stock `DataCollatorForLanguageModeling` builds labels from
    `input_ids` and would overwrite the mask this file exists to apply.
    `DataCollatorForSeq2Seq` happens to preserve labels, which is an
    implementation detail of a collator named for a different task.

    Pads on the right: this is training, so every position is attended in
    a single forward pass and the padded tail is excluded by
    `attention_mask` and by `-100` labels. (Generation is the case that
    needs left padding, and `hf_runner` generates one sequence at a time
    so it never pads at all.)
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[Example]) -> dict:
        import torch

        width = max(len(example.input_ids) for example in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for example in batch:
            padding = width - len(example.input_ids)

            input_ids.append(
                example.input_ids + [self.pad_token_id] * padding
            )
            labels.append(example.labels + [-100] * padding)
            attention_mask.append(
                [1] * len(example.input_ids) + [0] * padding
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }


def load_tokenizer(model_id: str) -> Any:
    """
    The tokenizer alone, so the data path can be checked before the GPU.

    Encoding 8,000 rows takes seconds and can refuse (a stale policy
    stamp, a row over the length budget). Loading a 4-bit 4B model takes
    minutes. Doing the cheap failing step first is the difference between
    finding out immediately and finding out after the weights are resident
    -- which on a session-limited Kaggle runtime is a real cost.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_for_training(
    model_id: str,
    *,
    lora_rank: int = 16,
    device_map: str | dict[str, int] = "auto",
) -> Any:
    """
    Load the base model 4-bit and wrap it in a LoRA adapter.

    `use_cache=False` is required with gradient checkpointing -- leaving
    it on makes `transformers` warn and silently disable checkpointing,
    which then OOMs on a T4 at a batch size that was chosen assuming
    checkpointing was active.

    `device_map` is the one argument that has to change under data
    parallelism, and getting it wrong is quiet rather than loud. Every
    rank sees every card, so `"auto"` makes *each* of two ranks shard one
    replica across *both* cards: four half-models on two devices, each
    rank's forward pass crossing the PCIe bus twice per layer, and both
    ranks contending for the same memory. It runs. It is several times
    slower than one card, and nothing in the logs says so. Under
    `torchrun` the caller passes `{"": local_rank}`, which is the whole
    replica pinned to this rank's own device.
    """

    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    dtype = pick_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGETS),
        ),
    )

    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.train_qlora",
        description=(
            "QLoRA fine-tune on data/train.jsonl, with checkpoint "
            "selection on data/validation.jsonl."
        ),
    )

    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("artifacts/qlora"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "stop after this many optimizer steps instead of completing "
            "--epochs. For calibration: a 20-step run on the real weights "
            "proves the chosen batch size fits both cards and measures "
            "throughput, before the full run is committed to. The adapter "
            "it writes is stamped `partial_run_max_steps` so it cannot be "
            "mistaken for a finished one."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "load the tokenizer, encode both splits, print the length "
            "distribution and the masking, then stop. Verifies the data "
            "path without a GPU."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    local_rank, _, world_size = ddp_world()

    announce("DriveMind QLoRA fine-tune")
    announce(f"  policy version  {POLICY_VERSION}")
    announce(f"  base model      {args.model}")
    announce(f"  adapter out     {args.out}")
    announce(
        f"  parallelism     {world_size} rank(s), "
        f"effective batch {args.batch_size * args.grad_accum * world_size}"
        f" = {args.batch_size} x {args.grad_accum} accum x {world_size}"
    )
    announce()

    train_rows = load_rows(args.data_dir / "train.jsonl")
    validation_rows = load_rows(args.data_dir / "validation.jsonl")

    tokenizer = load_tokenizer(args.model)

    train_set = encode_split(
        tokenizer,
        train_rows,
        name="train",
        max_length=args.max_length,
    )

    validation_set = encode_split(
        tokenizer,
        validation_rows,
        name="validation",
        max_length=args.max_length,
    )

    if args.dry_run:
        example = train_set[0]

        prompt_tokens = sum(
            1 for label in example.labels if label == -100
        )

        announce()
        announce(
            f"  masking check   {prompt_tokens} prompt tokens at -100, "
            f"{len(example.labels) - prompt_tokens} scored"
        )
        announce("  scored text     " + repr(
            tokenizer.decode(
                [
                    token
                    for token, label in zip(
                        example.input_ids,
                        example.labels,
                        strict=True,
                    )
                    if label != -100
                ]
            )
        ))
        announce("\n  dry run: nothing trained, nothing written.")

        return 0

    import torch
    from transformers import Trainer, TrainingArguments

    model = load_for_training(
        args.model,
        lora_rank=args.lora_rank,
        # One replica per rank, pinned to that rank's own card. See
        # `load_for_training` for why "auto" is wrong here and quiet
        # about being wrong.
        device_map="auto" if world_size == 1 else {"": local_rank},
    )

    dtype = pick_dtype()

    training_arguments = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        # -1 is "no cap, use --epochs". Anything above 0 wins over epochs.
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        # Non-reentrant checkpointing, which is not a style preference.
        # The reentrant implementation runs the recomputed forward outside
        # autograd's view, so DDP's parameter hooks never fire for the
        # checkpointed blocks and the all-reduce either hangs waiting for
        # gradients that will not arrive or raises about parameters marked
        # ready twice. Non-reentrant keeps the hooks.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Every trainable parameter here is a LoRA adapter and every one
        # of them is used in every forward pass -- the frozen base weights
        # are not registered with DDP at all. Leaving the search on costs
        # a full graph traversal per step to discover that.
        ddp_find_unused_parameters=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        # A float in [0, 1) is a *ratio* of total steps, an int is an exact
        # count. `warmup_ratio` was the old spelling and does not exist in
        # transformers 5.x -- passing it raises rather than warns.
        warmup_steps=0.03,
        optim="paged_adamw_8bit",
        # fp16 on Turing, bf16 on Ampere and later. Passing bf16=True on a
        # T4 is an error, not a slow path.
        fp16=dtype is torch.float16,
        bf16=dtype is torch.bfloat16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        # Selection on validation loss. `test.jsonl` is not loaded by this
        # file and must not be: a checkpoint chosen by test performance
        # has spent the test set.
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        # No experiment tracker. The corpus is derived from a filesystem
        # policy and the rows are the product's decision logic; shipping
        # them to a third-party service by default would be an upload
        # nobody asked for (§426).
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=Collator(tokenizer.pad_token_id),
    )

    trainer.train()

    # Rank 0 writes; the others return. Every rank holds an identical
    # adapter after `load_best_model_at_end`, so letting all of them write
    # would not corrupt the result by disagreement -- it would corrupt it
    # by two processes writing the same `.safetensors` at the same time.
    if not trainer.is_world_process_zero():
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    # Written next to the adapter because an adapter travels. Without it,
    # a directory of LoRA weights says nothing about which policy's labels
    # it learned or which base model it belongs on top of (§514/§516).
    record = {
        "base_model": args.model,
        "policy_version": POLICY_VERSION,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "epochs": args.epochs,
        # The product of all three, because the two ranks of a data-
        # parallel run contribute to one optimizer step. Recording
        # `batch * accum` here would understate it by the world size and
        # make two runs look like the same recipe when they are not.
        "world_size": world_size,
        "per_device_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": (
            args.batch_size * args.grad_accum * world_size
        ),
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_targets": list(LORA_TARGETS),
        "max_length": args.max_length,
        "precision": str(dtype).removeprefix("torch."),
        "seed": args.seed,
        "selection": "eval_loss on data/validation.jsonl",
    }

    if args.max_steps > 0:
        # A calibration adapter and a finished one are the same files in
        # the same layout. This is the only thing that distinguishes them,
        # so it is recorded rather than left to whoever remembers which
        # directory was which (§555).
        record["partial_run_max_steps"] = args.max_steps

    (args.out / "drivemind_run.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )

    announce(f"\n  adapter written to {args.out}")
    announce(
        "\n  Nothing has been measured yet. Training loss is not a metric "
        "anyone\n  should read; the four that matter come from the "
        "held-out sets, and\n  the base model has to be measured in this "
        "same environment for the\n  comparison to mean anything:\n"
        f"\n    python -m ml.eval_holdout --model {args.model}"
        f"\n    python -m ml.eval_holdout --model {args.model} "
        f"--adapter {args.out}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

