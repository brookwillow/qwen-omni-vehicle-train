#!/usr/bin/env python3
"""
Thinker-only LoRA training script for Qwen2.5-Omni with ModelScope Swift.

Trains only the language-thinker pathway for ReAct-style tool use,
freezing AUT/Talker/Vocoder and producing freeze audit artifacts.

Example:
  python train_thinker_lora.py \
    --model models/Qwen2.5-Omni-3B \
    --train-file data/train_final.jsonl \
    --output-dir ./lora_output
"""

from __future__ import annotations

import argparse
import importlib.util as importlib_util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
from datasets import load_dataset as hf_load_dataset
from peft import LoraConfig, get_peft_model
from swift import EncodePreprocessor, get_model_processor, get_template
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback


class SafeKeySeq2SeqTrainer(Seq2SeqTrainer):
    """Filter non-string keys from batch dicts (Swift M-RoPE position_ids workaround)."""

    def compute_loss(self, model, inputs, *args, **kwargs):
        clean = {k: v for k, v in inputs.items() if isinstance(k, str)}
        return super().compute_loss(model, clean, *args, **kwargs)


# Disable deepspeed probing (broken installs cause import errors).
if os.environ.get("SWIFT_DISABLE_DEEPSPEED", "1") == "1":
    _orig_find_spec = importlib_util.find_spec

    def _find_spec_without_deepspeed(name, *args, **kwargs):
        if name == "deepspeed":
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib_util.find_spec = _find_spec_without_deepspeed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thinker-only LoRA training for Qwen2.5-Omni")
    p.add_argument("--model", default="models/Qwen2.5-Omni-3B", help="Model path or id")
    p.add_argument("--model-type", default="qwen2_5_omni", help="Swift model_type")
    p.add_argument("--train-file", default="data/train_final.jsonl", help="Training JSONL (SP already injected)")
    p.add_argument("--output-dir", default="./lora_output", help="Output directory for checkpoints/adapters")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--do-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--forbidden-trainable-keywords",
        default="audio,talker,vocoder,audio_decoder,speech_decoder",
        help="Parameter names containing these keywords must not be trainable",
    )
    p.add_argument("--smoke-steps", type=int, default=0, help="If >0, override epochs for quick sanity run")
    return p.parse_args()


# ── Dataset helpers ──────────────────────────────────────────

def split_train_eval(train_ds, val_ratio: float, seed: int):
    split = train_ds.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def get_tokenizer(processor_or_tokenizer: Any):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def find_last_assistant_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def encode_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, add_special_tokens=False)
    return encoded["input_ids"]


def find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for idx in range(len(haystack) - len(needle), -1, -1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def build_last_assistant_labels(input_ids: list[int], target_ids: list[int]) -> tuple[list[int], bool]:
    labels = [-100] * len(input_ids)
    start = find_last_subsequence(input_ids, target_ids)
    if start < 0:
        return labels, False
    end = start + len(target_ids)
    labels[start:end] = input_ids[start:end]
    return labels, True


class LastAssistantLabelEncodePreprocessor(EncodePreprocessor):
    """Swift encoder that supervises only the final assistant message content."""

    def __init__(self, template, tokenizer):
        super().__init__(template=template)
        self.tokenizer = tokenizer

    def preprocess(self, row: dict) -> dict | None:
        target = find_last_assistant_content(row.get("messages") or [])
        encoded = super().preprocess(row)
        if encoded is None:
            return None

        input_ids = encoded.get("input_ids")
        if not isinstance(input_ids, list):
            return encoded

        labels = [-100] * len(input_ids)
        matched = False
        if target:
            target_ids = encode_text(self.tokenizer, target)
            if target_ids:
                labels, matched = build_last_assistant_labels(input_ids, target_ids)

        encoded["labels"] = labels
        encoded["label_matched"] = matched
        return encoded


def encode_dataset_with_last_assistant_labels(dataset_obj, template, tokenizer, split_name: str, num_proc: int):
    """Encode samples and supervise only the final assistant message content."""
    encoder = LastAssistantLabelEncodePreprocessor(template=template, tokenizer=tokenizer)
    encoded = encoder(dataset_obj, num_proc=num_proc)
    if "label_matched" in encoded.column_names:
        matched = sum(1 for value in encoded["label_matched"] if value)
        print(f"[labels] {split_name}: matched final assistant spans {matched}/{len(encoded)}")
    return encoded


def count_valid_label_tokens(example: dict) -> int:
    labels = example.get("labels") or []
    return sum(1 for label in labels if label != -100)


def filter_empty_label_examples(dataset_obj, split_name: str):
    """Drop examples that would produce NaN loss because all labels are ignored."""
    if "labels" not in dataset_obj.column_names:
        raise ValueError(f"{split_name} dataset has no labels column after encoding.")

    before = len(dataset_obj)
    dataset_obj = dataset_obj.filter(lambda example: count_valid_label_tokens(example) > 0)
    removed = before - len(dataset_obj)
    if removed:
        print(f"[warn] filtered {removed}/{before} {split_name} examples with no supervised label tokens.")
    else:
        print(f"[data] {split_name} label check: {before} examples, 0 empty-label examples.")
    if len(dataset_obj) == 0:
        raise ValueError(f"{split_name} dataset is empty after filtering empty-label examples.")
    return dataset_obj


# ── Freeze / audit helpers ───────────────────────────────────

def freeze_forbidden_params(model: torch.nn.Module, forbidden_keywords: Iterable[str]) -> int:
    frozen = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(kw and kw in lname for kw in forbidden_keywords):
            p.requires_grad = False
            frozen += 1
    return frozen


def summarize_trainable_params(
    model: torch.nn.Module,
    out_dir: Path,
    forbidden_keywords: Iterable[str],
) -> Tuple[Dict[str, int], int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trainable_file = out_dir / "trainable_params.txt"
    summary_file = out_dir / "freeze_summary.json"

    total_trainable = 0
    module_counter = {"aut": 0, "talker": 0, "vocoder": 0, "other": 0}
    forbidden_hits = []

    with trainable_file.open("w", encoding="utf-8") as f:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            total_trainable += n
            f.write(f"{name}\tshape={tuple(p.shape)}\tnumel={n}\n")

            lname = name.lower()
            if "aut" in lname or "audio" in lname:
                module_counter["aut"] += n
            elif "talker" in lname or "speech" in lname:
                module_counter["talker"] += n
            elif "vocoder" in lname:
                module_counter["vocoder"] += n
            else:
                module_counter["other"] += n

            for kw in forbidden_keywords:
                if kw and kw in lname:
                    forbidden_hits.append(name)
                    break

    summary = {
        "total_trainable_params": total_trainable,
        "by_module": module_counter,
        "forbidden_trainable_count": len(forbidden_hits),
        "forbidden_trainable_names_preview": forbidden_hits[:50],
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary, len(forbidden_hits)


# ── Trainer callback ─────────────────────────────────────────

class MetricsSaverCallback(TrainerCallback):
    """Append every logged metric step to {output_dir}/train_metrics.jsonl."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
        }
        for k in ("loss", "learning_rate", "grad_norm", "token_acc",
                  "eval_loss", "eval_token_acc"):
            if k in logs:
                record[k] = logs[k]
        if len(record) > 2:  # more than just step/epoch
            metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch is not None else None,
        }
        for k in ("eval_loss", "eval_token_acc"):
            if k in metrics:
                record[k] = metrics[k]
        if len(record) > 2:
            metrics_path = Path(args.output_dir) / "train_metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ConsoleMetricsCallback(TrainerCallback):
    @staticmethod
    def _fmt_metric(key, value):
        if not isinstance(value, float):
            return str(value)
        if key == "learning_rate":
            return f"{value:.6e}"
        return f"{value:.6f}"

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        parts = [
            f"{k}={self._fmt_metric(k, logs[k])}"
            for k in ("loss", "learning_rate", "grad_norm", "token_acc")
            if k in logs
        ]
        if parts:
            step, max_s = state.global_step, state.max_steps or "?"
            epoch = self._fmt_metric("epoch", state.epoch) if state.epoch is not None else "?"
            print(f"[train][step {step}/{max_s}][epoch {epoch}] " + " | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        parts = [
            f"{k}={self._fmt_metric(k, metrics[k])}"
            for k in ("eval_loss", "eval_token_acc")
            if k in metrics
        ]
        if parts:
            print(f"[eval][step {state.global_step}] " + " | ".join(parts))


def silence_default_trainer_console_callbacks(trainer) -> list[str]:
    """Remove default Trainer console callbacks so metrics print in one format."""
    handler = getattr(trainer, "callback_handler", None)
    callbacks = getattr(handler, "callbacks", None)
    if callbacks is None:
        return []

    muted_names = {"PrinterCallback", "ProgressCallback"}
    kept = []
    removed = []
    for callback in callbacks:
        name = callback.__class__.__name__
        if name in muted_names:
            removed.append(name)
        else:
            kept.append(callback)
    handler.callbacks = kept
    return removed


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.environ["NO_POSITION_IDS"] = "1"
    torch.manual_seed(args.seed)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print("[1/6] Loading model...")
    model, processor_or_tokenizer = get_model_processor(
        args.model,
        model_type=args.model_type,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=True,
    )
    try:
        template = get_template(processor_or_tokenizer)
    except Exception:
        template = get_template(getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer))
    if hasattr(template, "set_mode"):
        template.set_mode("train")
        print("[patch] Set template mode=train")
    if hasattr(template, "max_length"):
        template.max_length = args.max_length
        print(f"[patch] Set template.max_length={args.max_length}")

    # Patch broken _get_position_ids in some Swift versions (M-RoPE compat).
    if hasattr(template, "_get_position_ids"):
        _orig = template._get_position_ids

        def _safe_get_position_ids(res):
            try:
                result = _orig(res)
            except Exception:
                return {}
            return result if isinstance(result, dict) else {}

        template._get_position_ids = _safe_get_position_ids
        print("[patch] Wrapped template._get_position_ids")

    # 2. Load dataset
    print("[2/6] Loading dataset...")
    dataset = hf_load_dataset("json", data_files={"train": args.train_file})
    train_dataset, eval_dataset = split_train_eval(dataset["train"], args.val_ratio, args.seed)
    print(f"[data] train={len(train_dataset)} eval={len(eval_dataset)}")

    # 3. Encode
    print("[3/6] Encoding dataset...")
    tokenizer = get_tokenizer(processor_or_tokenizer)
    train_dataset = encode_dataset_with_last_assistant_labels(
        train_dataset, template, tokenizer, "train", args.num_proc
    )
    eval_dataset = encode_dataset_with_last_assistant_labels(
        eval_dataset, template, tokenizer, "eval", args.num_proc
    )
    train_dataset = filter_empty_label_examples(train_dataset, "train")
    eval_dataset = filter_empty_label_examples(eval_dataset, "eval")

    # Keep only model-relevant columns.
    MODEL_COLS = {"input_ids", "attention_mask", "labels", "position_ids"}
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in MODEL_COLS])
    eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c not in MODEL_COLS])
    print(f"[encode] columns: {train_dataset.column_names}")

    # 4. LoRA config
    print("[4/6] Building LoRA config...")
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        target_modules=target_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training arguments (compatible with different Swift/transformers versions).
    train_args_kwargs = {
        "output_dir": str(out_dir),
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "predict_with_generate": False,
        "bf16": (args.torch_dtype == "bfloat16"),
        "fp16": (args.torch_dtype == "float16"),
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": args.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_token_acc",
        "greater_is_better": True,
    }

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if args.do_eval:
        eval_key = "evaluation_strategy" if "evaluation_strategy" in sig else "eval_strategy"
        if eval_key in sig:
            train_args_kwargs[eval_key] = "epoch"
        if "do_eval" in sig:
            train_args_kwargs["do_eval"] = True
    else:
        for k in ("evaluation_strategy", "eval_strategy"):
            if k in sig:
                train_args_kwargs[k] = "no"
        if "do_eval" in sig:
            train_args_kwargs["do_eval"] = False

    filtered = {k: v for k, v in train_args_kwargs.items() if k in sig}
    train_args = Seq2SeqTrainingArguments(**filtered)
    if args.smoke_steps > 0:
        train_args.max_steps = args.smoke_steps
        train_args.num_train_epochs = 1

    # 6. Trainer
    print("[5/6] Initializing trainer...")
    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "template": template,
        "train_dataset": train_dataset,
    }
    if args.do_eval:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if "lora_config" in trainer_sig:
        trainer_kwargs["lora_config"] = lora_config
    else:
        model = get_peft_model(model, lora_config)
        trainer_kwargs["model"] = model

    trainer = SafeKeySeq2SeqTrainer(**trainer_kwargs)
    muted_callbacks = silence_default_trainer_console_callbacks(trainer)
    if muted_callbacks:
        print(f"[log] Disabled default Trainer console callbacks: {', '.join(muted_callbacks)}")
    trainer.add_callback(ConsoleMetricsCallback)
    trainer.add_callback(MetricsSaverCallback)

    # Freeze audit: ensure no audio/talker/vocoder params are trainable.
    forbidden_keywords = [x.strip().lower() for x in args.forbidden_trainable_keywords.split(",") if x.strip()]
    frozen = freeze_forbidden_params(trainer.model, forbidden_keywords)
    if frozen:
        print(f"[audit] Auto-froze {frozen} forbidden params")
    summary, forbidden_count = summarize_trainable_params(trainer.model, out_dir, forbidden_keywords)
    print(f"[audit] trainable={summary['total_trainable_params']:,} forbidden={forbidden_count}")
    print(f"[audit] by_module: {json.dumps(summary['by_module'])}")
    if forbidden_count > 0:
        raise RuntimeError(
            f"Found {forbidden_count} forbidden trainable params after freeze. "
            f"Check {out_dir / 'freeze_summary.json'}"
        )

    # Train
    print("[6/6] Start training...")
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Training complete. LoRA saved to: {out_dir}")


if __name__ == "__main__":
    main()
