#!/usr/bin/env python3
"""Run the canonical SFT -> eval -> DPO -> eval training pipeline.

The default mode is dry-run so updating this file is the single source of truth
for the current recommended commands without accidentally starting GPU jobs.
Pass ``--run`` to execute stages.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


STAGE_ALIASES = {
    "full": ["validate", "build", "sft", "eval-sft", "dpo-noise", "eval-dpo"],
    "sft-only": ["validate", "build", "sft", "eval-sft"],
    "dpo-only": ["dpo-noise", "eval-dpo"],
}

VALID_STAGES = {
    "validate",
    "build",
    "sft",
    "eval-sft",
    "dpo-noise",
    "eval-dpo",
}


@dataclass(frozen=True)
class StageCommand:
    name: str
    argv: list[str]
    log_dir: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    model_dir: str = "/home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B"
    train_file: str = "data/train_final.jsonl"
    eval_dir: str = "data/eval"
    system_prompt: str = "data/system-prompt.txt"
    sft_output_dir: str = "lora_output_sft_over_noise_repair"
    dpo_noise_output_dir: str = "lora_output_sft_over_noise_repair_dpo_noise"
    torch_dtype: str = "bfloat16"
    max_length: int = 4096
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_grad_accum: int = 16
    lora_r: int = 8
    lora_alpha: int = 16
    dpo_lr: str = "6e-7"
    dpo_beta: str = "0.05"
    dpo_epochs: int = 1
    dpo_batch_size: int = 1
    dpo_grad_accum: int = 8
    dpo_sft_loss_weight: str = "0.1"


def select_stages(stage: str) -> list[str]:
    if stage in STAGE_ALIASES:
        return STAGE_ALIASES[stage]
    if stage in VALID_STAGES:
        return [stage]
    raise ValueError(f"Unknown stage: {stage}")


def build_stage_commands(config: PipelineConfig, stages: list[str]) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for stage in stages:
        if stage == "validate":
            commands.append(StageCommand("validate-splits", ["python3", "scripts/validate_splits.py"]))
            commands.append(StageCommand("validate-rl-schema", ["python3", "scripts/validate_rl_schema.py"]))
        elif stage == "build":
            commands.append(
                StageCommand(
                    "build-train-data",
                    [
                        "python3",
                        "build_train_data.py",
                        "--sample-weight",
                        "hard_cases/OverNoiseRemaining_20260607.jsonl:4",
                        "hard_cases/*.jsonl:3",
                        "ProfileControl:2",
                        "WindowControl:1.5",
                        "CurrentNoiseWithHistoryTool:0.5",
                        "NoiseDoNotAct_coverage:0.5",
                        "--output",
                        config.train_file,
                    ],
                )
            )
        elif stage == "sft":
            commands.append(
                StageCommand(
                    "train-sft",
                    [
                        "python3",
                        "train_thinker_lora.py",
                        "--model",
                        config.model_dir,
                        "--train-file",
                        config.train_file,
                        "--output-dir",
                        config.sft_output_dir,
                        "--torch-dtype",
                        config.torch_dtype,
                        "--max-length",
                        str(config.max_length),
                        "--train-batch-size",
                        str(config.sft_batch_size),
                        "--grad-accum",
                        str(config.sft_grad_accum),
                        "--lora-r",
                        str(config.lora_r),
                        "--lora-alpha",
                        str(config.lora_alpha),
                        "--epochs",
                        str(config.sft_epochs),
                    ],
                    log_dir=Path(config.sft_output_dir),
                )
            )
        elif stage == "eval-sft":
            commands.append(
                StageCommand(
                    "eval-sft",
                    [
                        "python3",
                        "eval.py",
                        "batch",
                        "--model-dir",
                        config.model_dir,
                        "--lora-dir",
                        config.sft_output_dir,
                        "--eval-dir",
                        config.eval_dir,
                    ],
                    log_dir=Path(config.sft_output_dir),
                )
            )
        elif stage == "dpo-noise":
            commands.append(
                StageCommand(
                    "train-dpo-noise",
                    [
                        "python3",
                        "train_memory_dpo_lora.py",
                        "--model",
                        config.model_dir,
                        "--init-lora-dir",
                        config.sft_output_dir,
                        "--preference-file",
                        (
                            "data/rl/anti_over_noise_preferences.jsonl,"
                            "data/rl/noise_false_positive_preferences.jsonl,"
                            "data/rl/still_over_noise_preferences_round3.jsonl"
                        ),
                        "--preference-weight",
                        "anti_over_noise_preferences.jsonl:4",
                        "still_over_noise_preferences_round3.jsonl:6",
                        "--output-dir",
                        config.dpo_noise_output_dir,
                        "--prompt-format",
                        "chat_template",
                        "--system-prompt",
                        config.system_prompt,
                        "--lr",
                        config.dpo_lr,
                        "--beta",
                        config.dpo_beta,
                        "--epochs",
                        str(config.dpo_epochs),
                        "--train-batch-size",
                        str(config.dpo_batch_size),
                        "--grad-accum",
                        str(config.dpo_grad_accum),
                        "--sft-loss-weight",
                        config.dpo_sft_loss_weight,
                        "--reference-mode",
                        "reference_free",
                    ],
                    log_dir=Path(config.dpo_noise_output_dir),
                )
            )
        elif stage == "eval-dpo":
            commands.append(
                StageCommand(
                    "eval-dpo-noise",
                    [
                        "python3",
                        "eval.py",
                        "batch",
                        "--model-dir",
                        config.model_dir,
                        "--lora-dir",
                        config.dpo_noise_output_dir,
                        "--eval-dir",
                        config.eval_dir,
                    ],
                    log_dir=Path(config.dpo_noise_output_dir),
                )
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
    return commands


def shell_join(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def append_log(log_dir: Path | None, line: str) -> None:
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_commands(commands: list[StageCommand], *, dry_run: bool) -> None:
    started = datetime.now().isoformat(timespec="seconds")
    for index, command in enumerate(commands, start=1):
        command_line = shell_join(command.argv)
        header = f"[{index}/{len(commands)}] {command.name}: {command_line}"
        print(header)
        append_log(command.log_dir, f"{started} {header}")
        if dry_run:
            continue
        subprocess.run(command.argv, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=sorted(VALID_STAGES | set(STAGE_ALIASES)),
        help="Pipeline stage to print or run.",
    )
    parser.add_argument("--run", action="store_true", help="Execute commands. Default is dry-run.")
    parser.add_argument("--model-dir", default=PipelineConfig.model_dir)
    parser.add_argument("--train-file", default=PipelineConfig.train_file)
    parser.add_argument("--eval-dir", default=PipelineConfig.eval_dir)
    parser.add_argument("--system-prompt", default=PipelineConfig.system_prompt)
    parser.add_argument("--sft-output-dir", default=PipelineConfig.sft_output_dir)
    parser.add_argument("--dpo-noise-output-dir", default=PipelineConfig.dpo_noise_output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        model_dir=args.model_dir,
        train_file=args.train_file,
        eval_dir=args.eval_dir,
        system_prompt=args.system_prompt,
        sft_output_dir=args.sft_output_dir,
        dpo_noise_output_dir=args.dpo_noise_output_dir,
    )
    stages = select_stages(args.stage)
    commands = build_stage_commands(config, stages)
    run_commands(commands, dry_run=not args.run)
    if not args.run:
        print("\n[dry-run] Add --run to execute these commands.")


if __name__ == "__main__":
    main()
