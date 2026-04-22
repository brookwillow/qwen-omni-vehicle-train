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
from typing import Dict, Iterable, Tuple

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


def ensure_labels_column(dataset_obj, im_start_id: int = 151644, im_end_id: int = 151645):
    """Build assistant-only loss mask from input_ids if labels column is missing."""
    if "labels" in dataset_obj.column_names:
        return dataset_obj
    if "input_ids" not in dataset_obj.column_names:
        raise ValueError(f"Dataset has no labels/input_ids. Columns: {dataset_obj.column_names}")

    print("[warn] labels column missing; building assistant-only masked labels from input_ids.")

    def _build_masked_labels(example):
        input_ids = example["input_ids"]
        labels = [-100] * len(input_ids)
        n = len(input_ids)
        i = 0
        while i < n:
            if input_ids[i] == im_start_id:
                role_end = None
                is_assistant = False
                for j in range(i + 1, min(i + 7, n)):
                    if input_ids[j] in (198, 271):  # '\n' tokens
                        role_end = j
                        break
                if role_end is not None:
                    if 77091 in input_ids[i + 1 : role_end]:  # 77091 == 'assistant'
                        is_assistant = True
                if is_assistant and role_end is not None:
                    content_start = role_end + 1
                    content_end = n
                    for j in range(content_start, n):
                        if input_ids[j] == im_end_id:
                            content_end = j + 1
                            break
                    for j in range(content_start, content_end):
                        labels[j] = input_ids[j]
                    i = content_end
                    continue
            i += 1
        example["labels"] = labels
        return example

    return dataset_obj.map(_build_masked_labels)


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

class ConsoleMetricsCallback(TrainerCallback):
    @staticmethod
    def _fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        parts = [f"{k}={self._fmt(logs[k])}" for k in ("loss", "learning_rate", "grad_norm", "token_acc") if k in logs]
        if parts:
            step, max_s = state.global_step, state.max_steps or "?"
            epoch = self._fmt(state.epoch) if state.epoch is not None else "?"
            print(f"[train][step {step}/{max_s}][epoch {epoch}] " + " | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        parts = [f"{k}={self._fmt(metrics[k])}" for k in ("eval_loss", "eval_token_acc") if k in metrics]
        if parts:
            print(f"[eval][step {state.global_step}] " + " | ".join(parts))


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
    encoder_sig = inspect.signature(EncodePreprocessor.__init__).parameters
    if "max_length" in encoder_sig:
        encoder = EncodePreprocessor(template=template, max_length=args.max_length)
    else:
        encoder = EncodePreprocessor(template=template)
    train_dataset = encoder(train_dataset, num_proc=args.num_proc)
    eval_dataset = encoder(eval_dataset, num_proc=args.num_proc)

    train_dataset = ensure_labels_column(train_dataset)
    eval_dataset = ensure_labels_column(eval_dataset)

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
    trainer.add_callback(ConsoleMetricsCallback)

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
