"""
The bridge from a Hugging Face causal LM to DriveMind's evaluator.

This module is deliberately **not** inside `backend/`.
`tests/test_read_only.py::test_only_the_api_layer_depends_on_a_third_party_package`
fails if anything below the dev API imports a non-stdlib package, and that
guarantee is worth more than the convenience of putting a `transformers`
import next to the other providers. The decision core does not need a model
to work; a model runtime is an adapter that sits outside it.

What this file is careful about, because each one is a way to accidentally
report a better number than the model earned:

*   **The prompt comes from `build_ai_prompt`.** Not a copy of it, not a
    reworded version. The model is evaluated on exactly the string the
    corpus was built from, or the measurement is of a different task.
*   **The raw text is returned unmodified.** No fence stripping, no
    "extract the first JSON object", no repair. `parse_ai_response` is the
    contract; a model that wraps its answer in ```json is a model that
    failed the contract, and pre-cleaning its output moves that failure out
    of the metric and into this file where nobody will see it.
*   **Thinking mode is off by default** for chat templates that support it,
    and recorded in the run configuration. Qwen3 emits `<think>` blocks
    when it is on, which the strict parser rejects -- configuring the model
    in its documented non-thinking mode is using it correctly, but leaving
    that choice implicit would make two runs incomparable.

Usage, on a machine with the weights and a GPU:

    from ml.hf_runner import load_model, make_emitter
    from backend.services.ai_holdout import (
        evaluate_emitter, format_held_out_report,
    )

    bundle = load_model("Qwen/Qwen3-4B", load_in_4bit=True)
    results, verified = evaluate_emitter(make_emitter(bundle))
    print(format_held_out_report(results, bundle.describe(), verified))
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.models.system import AICase
from backend.services.ai_prompt import build_ai_prompt

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class GenerationConfig:
    """
    How the model is sampled. Greedy, and the reason is reproducibility.

    `do_sample=False` means two runs of the same model over the same
    corpus produce the same numbers. With sampling on, a re-run that
    disagrees with the last one is indistinguishable from a regression,
    and "structured-output validity 94%" becomes a figure with an
    unstated confidence interval. Sampling is available because a
    temperature sweep is a legitimate experiment -- it is just not the
    default for a reported metric.

    `max_new_tokens` is a budget, not a target. The contract response is
    roughly 60-100 tokens; 256 leaves room for a verbose explanation
    without letting a looping model cost an hour. A truncated response
    fails to parse, which is the correct outcome: the model did not
    produce a valid answer within a bound a local runtime can afford.
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 42

    def describe(self) -> str:
        if not self.do_sample:
            return f"greedy, max_new_tokens={self.max_new_tokens}"

        return (
            f"sampled(temperature={self.temperature}, top_p={self.top_p}, "
            f"seed={self.seed}), max_new_tokens={self.max_new_tokens}"
        )

def render_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """
    Render one DriveMind prompt into the model's chat format.

    **This function is the train/serve skew control (§78).** The
    fine-tune formats its examples by calling this, and the evaluator
    formats its cases by calling this, so there is one definition of what
    the model sees. Two separate formatters that agree today are two
    formatters that will disagree after someone edits one of them, and
    the failure mode is a model that scores well on its own training
    format and badly in production.

    `build_ai_prompt` already contains the system instruction, so the
    whole thing goes in as a single user turn rather than being split
    across a `system` and a `user` message. Splitting it would be more
    idiomatic and would change the string the corpus was built from,
    which matters more.

    `enable_thinking` is passed through when the template accepts it.
    Templates that do not take the argument raise `TypeError`, so it is
    tried and then dropped rather than probed -- the set of templates
    that support it is not knowable in advance.
    """

    messages = [{"role": "user", "content": prompt}]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

def render_target(response: dict) -> str:
    """
    Render a training row's `response` into the exact string the model is
    asked to produce.

    Key order follows the contract as it is written in
    `SYSTEM_INSTRUCTION` -- action, risk, explanation -- and separators
    are compact, because a fine-tune learns the surface form it is shown.
    Training on `indent=2` JSON and then measuring a model that emits
    compact JSON would still parse, but training on a *different* key
    order than the prompt specifies teaches the model to ignore the
    prompt.

    Only the three contract fields are emitted. A row's `case_id` and
    `policy_version` are deliberately dropped: they sit outside
    `response` in the corpus precisely so they are never mistaken for
    something the model should produce.
    """

    return json.dumps(
        {
            "action": response["action"],
            "risk": response["risk"],
            "explanation": response["explanation"],
        },
        ensure_ascii=False,
        separators=(", ", ": "),
    )


@dataclass
class ModelBundle:
    """A loaded model, its tokenizer, and how it was configured."""

    model: Any
    tokenizer: Any
    model_id: str
    quantization: str
    enable_thinking: bool
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    adapter_path: str | None = None

    def describe(self) -> str:
        """
        The provider line that goes into the report header.

        Everything in here changes the numbers, so all of it is in the
        one string the report prints. A held-out figure recorded against
        "Qwen3-4B" is not comparable with another figure recorded against
        "Qwen3-4B" if one was 4-bit with thinking on and the other was
        bf16 with thinking off (§514/§516).
        """

        parts = [self.model_id, self.quantization]

        if self.adapter_path:
            parts.append(f"lora={self.adapter_path}")

        parts.append(
            "thinking=on" if self.enable_thinking else "thinking=off"
        )
        parts.append(self.generation.describe())

        return "  ".join(parts)

def pick_dtype() -> Any:
    """
    bf16 where the hardware has it, fp16 where it does not.

    Not cosmetic. The target training environment is 2x Tesla T4, which
    is Turing (compute capability 7.5) and has **no bfloat16 support** --
    a hardcoded `torch.bfloat16` either errors or silently falls back to
    an emulated path that is slower than fp16. The local RTX 3050 is
    Ampere and does have bf16, so a dtype that works on the development
    machine is exactly the kind of choice that breaks only on the machine
    that matters.

    Returns a torch dtype, so this is only callable where torch is
    installed.
    """

    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    load_in_4bit: bool = True,
    adapter_path: str | None = None,
    enable_thinking: bool = False,
    generation: GenerationConfig | None = None,
    device_map: str = "auto",
) -> ModelBundle:
    """
    Load a causal LM for evaluation, optionally with a LoRA adapter.

    `transformers` is imported here rather than at module scope so that
    reading this file, or importing `render_prompt` / `render_target` from
    it, does not require a multi-gigabyte install. The formatting
    functions are the part `train_qlora` needs to share, and they are
    pure stdlib.

    4-bit NF4 with double quantization is the default because the target
    hardware is a 4 GB laptop GPU; on a 16 GB accelerator, pass
    `load_in_4bit=False` for bf16. The choice is recorded in
    `ModelBundle.describe()` because it changes the numbers.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = pick_dtype()

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "dtype": dtype,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # The compute dtype is in the label because it is part of the
    # measurement: 4-bit NF4 with an fp16 compute path on a T4 and the
    # same quantization with bf16 on an Ampere card are not guaranteed to
    # produce the same tokens.
    precision = str(dtype).removeprefix("torch.")

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        quantization=(
            f"4bit-nf4/{precision}" if load_in_4bit else precision
        ),
        enable_thinking=enable_thinking,
        generation=generation or GenerationConfig(),
        adapter_path=adapter_path,
    )

def generate(bundle: ModelBundle, prompt: str) -> str:
    """
    Generate one continuation and return it **unmodified**.

    The two things this does not do are the point:

    *   It decodes only the newly generated tokens. Returning the prompt
        echo as well would fail every parse for a reason that has nothing
        to do with the model.
    *   It does not clean the result. No ```json fence stripping, no
        "find the first `{`", no trailing-comma repair. Each of those
        would raise structured-output validity without the model having
        got any better at the contract, and the repaired case would be
        counted as a success (§404). `parse_ai_response` is the contract
        and it is strict on purpose.

    `skip_special_tokens=True` is a decode setting, not cleaning: it
    drops the chat framing (`<|im_end|>` and friends), which is transport
    and not the model's answer. Note what this does *not* rescue -- a
    Qwen3 model in thinking mode loses the `<think>` tags but keeps the
    reasoning text, so the response still fails to parse, which is the
    honest result for a model configured to answer in prose.
    """

    import torch

    text = render_prompt(
        bundle.tokenizer,
        prompt,
        enable_thinking=bundle.enable_thinking,
    )

    inputs = bundle.tokenizer(text, return_tensors="pt").to(
        bundle.model.device
    )

    config = bundle.generation

    if config.do_sample:
        from transformers import set_seed

        set_seed(config.seed)

    with torch.inference_mode():
        outputs = bundle.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature if config.do_sample else None,
            top_p=config.top_p if config.do_sample else None,
            pad_token_id=bundle.tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    return bundle.tokenizer.decode(generated, skip_special_tokens=True)


def make_emitter(bundle: ModelBundle) -> Callable[[AICase], str]:
    """
    Adapt a loaded model to the interface `evaluate_emitter` takes.

    One case per call, which is what the evaluator's interface is: the
    held-out sets are consumed lazily, so there is no batch to form
    without changing `backend`. On a 4 GB laptop GPU batch size 1 is the
    only size that fits anyway; on a larger accelerator this is the
    throughput ceiling to raise first, and 1,126 sequential generations
    are the reason a full held-out run is measured in tens of minutes
    rather than seconds.

    The prompt is `build_ai_prompt(case)` -- the same function the corpus
    was written from, not a copy of its output.
    """

    def emit(case: AICase) -> str:
        return generate(bundle, build_ai_prompt(case))

    return emit

