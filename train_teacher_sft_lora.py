#!/usr/bin/env python3
"""SFT LoRA training for a Qwen3.5 27B text teacher.

The teacher uses the same ``data/train_final.jsonl`` produced for Qwen Omni:
the system prompt is already injected by ``build_train_data.py`` and only the
final assistant answer after the current user turn is supervised.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


DEFAULT_SAMPLE_WEIGHTS = (
    "hard_cases/OverNoiseRemaining_20260607.jsonl:4",
    "hard_cases/*.jsonl:3",
    "ProfileControl:2",
    "WindowControl:1.5",
    "CurrentNoiseWithHistoryTool:0.5",
    "NoiseDoNotAct_coverage:0.5",
)
DEFAULT_MODELSCOPE_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_MODELSCOPE_MODEL_DIR = "/home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen3.5-27B"
CHAT_TEMPLATE_FALLBACK_WARNED = False


@dataclass(frozen=True)
class TeacherSftConfig:
    train_file: str = "data/train_final.jsonl"
    sample_weights: tuple[str, ...] = DEFAULT_SAMPLE_WEIGHTS


class _TeeStream:
    """Mirror stdout/stderr to both console and a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def setup_run_logs(output_dir: Path) -> TextIO:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "teacher_train_metrics.jsonl"
    train_log_path = output_dir / "teacher_train.log"
    metrics_path.write_text("", encoding="utf-8")
    train_log = train_log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, train_log)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, train_log)  # type: ignore[assignment]
    print(f"[log] metrics reset: {metrics_path}")
    print(f"[log] console log: {train_log_path}")
    return train_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-27B teacher SFT LoRA training")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELSCOPE_MODEL_DIR,
        help="Local Qwen3.5 teacher model path. Defaults to the ModelScope local_dir.",
    )
    parser.add_argument(
        "--modelscope-model",
        default=DEFAULT_MODELSCOPE_MODEL,
        help="ModelScope model id used when --download-modelscope is enabled.",
    )
    parser.add_argument(
        "--modelscope-cache-dir",
        default="",
        help="Optional ModelScope cache_dir passed to `modelscope download`.",
    )
    parser.add_argument(
        "--download-modelscope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download --modelscope-model to --model if the local model directory is missing.",
    )
    parser.add_argument("--train-file", default="data/train_final.jsonl", help="SP-injected chat JSONL")
    parser.add_argument("--system-prompt", default="data/system-prompt.txt", help="Documented SP source")
    parser.add_argument("--output-dir", default="teacher_lora_qwen35_27b_sft", help="LoRA output dir")
    parser.add_argument("--rebuild-train-data", action="store_true", help="Run build_train_data.py first")
    parser.add_argument(
        "--sample-weight",
        nargs="*",
        default=list(DEFAULT_SAMPLE_WEIGHTS),
        help="Weights passed to build_train_data.py when --rebuild-train-data is used",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit QLoRA instead of BF16 LoRA")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vision-keywords",
        default="vision,visual,image,patch_embed,merger",
        help="Comma-separated parameter name fragments to freeze after LoRA injection",
    )
    parser.add_argument("--smoke-steps", type=int, default=0, help="If >0, run only this many optimizer steps")
    return parser.parse_args(argv)


def build_build_data_command(config: TeacherSftConfig) -> list[str]:
    return [
        "python3",
        "build_train_data.py",
        "--sample-weight",
        *config.sample_weights,
        "--output",
        config.train_file,
    ]


def build_modelscope_download_command(args: argparse.Namespace) -> list[str]:
    command = ["modelscope", "download", "--model", args.modelscope_model]
    if args.modelscope_cache_dir:
        command.extend(["--cache_dir", args.modelscope_cache_dir])
    command.extend(["--local_dir", args.model])
    return command


def resolve_model_path(args: argparse.Namespace) -> str:
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        return str(model_path)
    if not args.download_modelscope:
        return args.model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_modelscope_download_command(args)
    print("[modelscope] local model not found; downloading:", " ".join(command))
    subprocess.run(command, check=True)
    return str(model_path)


def normalize_chatml_text(text: str) -> str:
    return text.rstrip()


def render_chatml_messages(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(chunks)


def find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    for idx in range(len(haystack) - len(needle), -1, -1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return -1


def build_label_mask(input_ids: list[int], target_ids: list[int]) -> tuple[list[int], bool]:
    labels = [-100] * len(input_ids)
    start = find_last_subsequence(input_ids, target_ids)
    if start < 0:
        return labels, False
    end = start + len(target_ids)
    labels[start:end] = input_ids[start:end]
    return labels, True


def find_last_assistant_content(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def find_final_assistant_after_last_user(messages: list[dict[str, Any]]) -> str:
    last_user_idx = -1
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = idx
    if last_user_idx < 0:
        return ""
    for msg in reversed(messages[last_user_idx + 1 :]):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def require(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install the training deps in qwen-omni first."
        ) from exc


def dtype_from_name(torch_module, name: str):
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def encode_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    global CHAT_TEMPLATE_FALLBACK_WARNED
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("Each row must contain a list-valued 'messages' field.")

    target = find_final_assistant_after_last_user(messages) or find_last_assistant_content(messages)
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        if not CHAT_TEMPLATE_FALLBACK_WARNED:
            print(f"[warn] tokenizer chat_template failed; using ChatML fallback: {exc}")
            CHAT_TEMPLATE_FALLBACK_WARNED = True
        rendered = render_chatml_messages(messages)
    rendered = normalize_chatml_text(rendered)
    encoded = tokenizer(rendered, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = list(encoded["input_ids"])

    target_ids = tokenizer.encode(target, add_special_tokens=False) if target else []
    labels, matched = build_label_mask(input_ids, target_ids)
    encoded["labels"] = labels
    encoded["label_matched"] = bool(matched)
    encoded["valid_label_tokens"] = sum(1 for label in labels if label != -100)
    return encoded


def build_collator(tokenizer: Any):
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        feature_copies = [dict(feature) for feature in features]
        label_features = [feature.pop("labels") for feature in feature_copies]
        batch = tokenizer.pad(feature_copies, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for labels in label_features:
            pad_len = max_len - len(labels)
            padded_labels.append(labels + [-100] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return collate


def write_trainable_audit(model: Any, output_dir: Path) -> None:
    trainable_file = output_dir / "teacher_trainable_params.txt"
    summary_file = output_dir / "teacher_lora_summary.json"
    total = 0
    with trainable_file.open("w", encoding="utf-8") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            count = param.numel()
            total += count
            f.write(f"{name}\tshape={tuple(param.shape)}\tnumel={count}\n")
    summary_file.write_text(
        json.dumps({"total_trainable_params": total}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[audit] trainable={total:,}")


def freeze_trainable_by_keywords(model: Any, keywords: list[str]) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        if any(keyword and keyword in lower_name for keyword in keywords):
            param.requires_grad = False
            frozen += 1
    return frozen


def build_metrics_callback(transformers: Any):
    class MetricsSaverCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            record = {"step": state.global_step, "epoch": state.epoch}
            record.update(logs)
            if len(record) <= 2:
                return
            metrics_path = Path(args.output_dir) / "teacher_train_metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return MetricsSaverCallback()


def build_training_arguments(transformers: Any, args: argparse.Namespace):
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
        "remove_unused_columns": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "seed": args.seed,
    }
    sig = inspect.signature(transformers.TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    if eval_key in sig:
        kwargs[eval_key] = args.eval_strategy
    if args.smoke_steps > 0 and "max_steps" in sig:
        kwargs["max_steps"] = args.smoke_steps
    return transformers.TrainingArguments(**{k: v for k, v in kwargs.items() if k in sig})


def load_teacher_model(transformers: Any, args: argparse.Namespace, dtype: Any, quantization_config: Any):
    kwargs = {
        "torch_dtype": dtype if not args.load_in_4bit else None,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        "quantization_config": quantization_config,
    }
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    except Exception as causal_exc:
        image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if image_text_cls is None:
            raise
        print(
            "[model] AutoModelForCausalLM failed; falling back to "
            f"AutoModelForImageTextToText ({causal_exc.__class__.__name__}: {causal_exc})"
        )
        return image_text_cls.from_pretrained(args.model, **kwargs)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    setup_run_logs(output_dir)
    model_path = resolve_model_path(args)

    if args.rebuild_train_data:
        command = build_build_data_command(
            TeacherSftConfig(train_file=args.train_file, sample_weights=tuple(args.sample_weight))
        )
        print("[data] rebuilding train file:", " ".join(command))
        subprocess.run(command, check=True)

    torch = require("torch")
    datasets = require("datasets")
    transformers = require("transformers")
    peft = require("peft")

    torch.manual_seed(args.seed)
    dtype = dtype_from_name(torch, args.torch_dtype)
    compute_dtype = dtype_from_name(torch, args.bnb_4bit_compute_dtype)

    print(f"[1/5] loading tokenizer: {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2/5] loading dataset")
    dataset = datasets.load_dataset("json", data_files={"train": args.train_file})["train"]
    split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"[data] train={len(train_dataset)} eval={len(eval_dataset)}")

    print("[3/5] encoding dataset")
    encode_fn = lambda row: encode_row(row, tokenizer, args.max_length)
    train_dataset = train_dataset.map(encode_fn, num_proc=args.num_proc)
    eval_dataset = eval_dataset.map(encode_fn, num_proc=args.num_proc)
    train_matched = sum(1 for value in train_dataset["label_matched"] if value)
    eval_matched = sum(1 for value in eval_dataset["label_matched"] if value)
    print(f"[labels] train matched final assistant spans {train_matched}/{len(train_dataset)}")
    print(f"[labels] eval matched final assistant spans {eval_matched}/{len(eval_dataset)}")
    train_dataset = train_dataset.filter(lambda row: row["valid_label_tokens"] > 0)
    eval_dataset = eval_dataset.filter(lambda row: row["valid_label_tokens"] > 0)
    keep_columns = {"input_ids", "attention_mask", "labels"}
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in keep_columns])
    eval_dataset = eval_dataset.remove_columns([c for c in eval_dataset.column_names if c not in keep_columns])

    print(f"[4/5] loading model: {model_path}")
    args.model = model_path
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = load_teacher_model(transformers, args, dtype, quantization_config)
    model.config.use_cache = False
    if args.load_in_4bit:
        model = peft.prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    lora_config = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = peft.get_peft_model(model, lora_config)
    if args.freeze_vision:
        vision_keywords = [item.strip().lower() for item in args.vision_keywords.split(",") if item.strip()]
        frozen = freeze_trainable_by_keywords(model, vision_keywords)
        if frozen:
            print(f"[audit] auto-froze {frozen} vision-related trainable params")
    write_trainable_audit(model, output_dir)

    print("[5/5] training")
    training_args = build_training_arguments(transformers, args)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if args.eval_strategy != "no" else None,
        data_collator=build_collator(tokenizer),
        callbacks=[build_metrics_callback(transformers)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Training complete. Teacher LoRA saved to: {output_dir}")


if __name__ == "__main__":
    main()
