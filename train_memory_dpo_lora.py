#!/usr/bin/env python3
"""Continue training an existing SFT LoRA with memory preference data.

This script implements a small DPO-style loop for the memory-use experiment.
It starts from ``base model + --init-lora-dir`` and writes a new adapter to
``--output-dir``.  The source SFT adapter is never modified.

Input JSONL rows can either contain an explicit prompt:
  {"prompt": "...", "chosen": "...", "rejected": "..."}

or a memory-task shape whose chosen/rejected values are final assistant outputs:
  {"history": [...], "current_query": "...", "chosen": {...}, "rejected": {...}}
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO


@dataclass
class PreferenceRow:
    prompt: str
    chosen: str
    rejected: str
    history: list[dict[str, str]]
    current_query: str
    task_type: str = ""


class _TeeStream:
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
    metrics_path = output_dir / "dpo_metrics.jsonl"
    train_log_path = output_dir / "dpo_train.log"
    metrics_path.write_text("", encoding="utf-8")
    train_log = train_log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, train_log)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, train_log)  # type: ignore[assignment]
    print(f"[log] metrics reset: {metrics_path}")
    print(f"[log] console log: {train_log_path}")
    return train_log


def normalize_response(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def format_memory_prompt(row: dict[str, Any]) -> str:
    payload = {
        "history": row.get("history", []),
        "current_query": row.get("current_query", row.get("query", "")),
    }
    if row.get("task_type"):
        payload["task_type"] = row["task_type"]
    return (
        "以下是车载语音助手的多轮对话历史和当前用户请求。"
        "请给出当前轮最终 assistant 输出；如果需要工具调用，只输出一行紧凑工具 JSON；"
        "如果需要澄清，直接输出中文追问；如果应拒识，仅输出 Reject。\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        "assistant:"
    )


def format_memory_messages(row: PreferenceRow, system_prompt: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    for msg in row.history:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    if row.current_query:
        messages.append({"role": "user", "content": row.current_query})
    return messages


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"chat template output dict missing input_ids: {value.keys()}")
        value = value["input_ids"]
    elif hasattr(value, "data") and isinstance(value.data, Mapping) and "input_ids" in value.data:
        value = value.data["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"expected single chat template sequence, got batch size {len(value)}")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token_id, int) for token_id in value):
        raise ValueError(f"chat template output must resolve to list[int], got {type(value).__name__}")
    return value


def load_preference_rows(path: str | Path, max_samples: int = 0) -> list[PreferenceRow]:
    rows: list[PreferenceRow] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            prompt = raw.get("prompt") or format_memory_prompt(raw)
            chosen = raw.get("chosen", raw.get("chosen_output"))
            rejected = raw.get("rejected", raw.get("rejected_output"))
            if chosen is None or rejected is None:
                raise ValueError(f"{path}:{line_no} missing chosen/rejected fields")
            chosen_text = normalize_response(chosen)
            rejected_text = normalize_response(rejected)
            if chosen_text == rejected_text:
                raise ValueError(f"{path}:{line_no} chosen and rejected are identical")
            rows.append(
                PreferenceRow(
                    prompt=str(prompt).strip(),
                    chosen=chosen_text,
                    rejected=rejected_text,
                    history=[
                        {"role": str(msg.get("role", "")), "content": str(msg.get("content", ""))}
                        for msg in raw.get("history", [])
                        if isinstance(msg, dict)
                    ],
                    current_query=str(raw.get("current_query", raw.get("query", ""))),
                    task_type=str(raw.get("task_type", "")),
                )
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No preference rows loaded from {path}")
    return rows


def expand_preference_files(value: str) -> list[Path]:
    paths: list[Path] = []
    for part in value.split(","):
        pattern = part.strip()
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(item) for item in matches)
        else:
            paths.append(Path(pattern))
    if not paths:
        raise ValueError("No preference files configured")
    return paths


def load_preference_rows_many(value: str, max_samples: int = 0) -> list[PreferenceRow]:
    rows: list[PreferenceRow] = []
    for path in expand_preference_files(value):
        remaining = max_samples - len(rows) if max_samples > 0 else 0
        if max_samples > 0 and remaining <= 0:
            break
        rows.extend(load_preference_rows(path, remaining))
    if not rows:
        raise ValueError(f"No preference rows loaded from {value}")
    return rows


def split_train_eval(rows: list[PreferenceRow], val_ratio: float, seed: int) -> tuple[list[PreferenceRow], list[PreferenceRow]]:
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if val_ratio <= 0:
        return rows, []
    val_size = max(1, int(len(rows) * val_ratio))
    return rows[val_size:], rows[:val_size]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO-style memory preference training from an existing SFT LoRA.")
    p.add_argument("--model", default="models/Qwen2.5-Omni-3B", help="Base model path or id")
    p.add_argument("--model-type", default="qwen2_5_omni", help="Swift model_type")
    p.add_argument("--init-lora-dir", required=True, help="Existing SFT LoRA adapter used as trainable initialization")
    p.add_argument("--reference-lora-dir", default="", help="Frozen reference adapter. Defaults to --init-lora-dir when --reference-mode=frozen_init")
    p.add_argument(
        "--preference-file",
        default="data/rl/memory_preferences.jsonl",
        help="Preference JSONL file, comma-separated files, or glob pattern.",
    )
    p.add_argument("--output-dir", default="./lora_output_dpo_memory")
    p.add_argument("--device-map", default="cuda:0")
    p.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--prompt-format", choices=["chat_template", "json_instruction"], default="chat_template")
    p.add_argument("--system-prompt", default="data/system-prompt.txt")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--sft-loss-weight", type=float, default=0.0, help="Optional NLL loss on chosen response")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=0, help="0 = save only at end")
    p.add_argument("--reference-mode", choices=["reference_free", "frozen_init"], default="reference_free")
    p.add_argument("--forbidden-trainable-keywords", default="audio,talker,vocoder,audio_decoder,speech_decoder")
    return p.parse_args()


def _lazy_import_training_deps():
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from swift import get_model_processor
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup

    return torch, F, PeftModel, get_model_processor, DataLoader, Dataset, get_cosine_schedule_with_warmup


def get_tokenizer(processor_or_tokenizer: Any):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def freeze_forbidden_params(model: Any, forbidden_keywords: Iterable[str]) -> int:
    frozen = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if any(kw and kw in lname for kw in forbidden_keywords):
            p.requires_grad = False
            frozen += 1
    return frozen


def summarize_trainable_params(model: Any, out_dir: Path, forbidden_keywords: Iterable[str]) -> dict[str, Any]:
    total_trainable = 0
    forbidden_hits = []
    trainable_file = out_dir / "dpo_trainable_params.txt"
    with trainable_file.open("w", encoding="utf-8") as f:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            total_trainable += p.numel()
            f.write(f"{name}\tshape={tuple(p.shape)}\tnumel={p.numel()}\n")
            lname = name.lower()
            if any(kw and kw in lname for kw in forbidden_keywords):
                forbidden_hits.append(name)
    summary = {
        "total_trainable_params": total_trainable,
        "forbidden_trainable_count": len(forbidden_hits),
        "forbidden_trainable_names_preview": forbidden_hits[:50],
    }
    (out_dir / "dpo_freeze_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def load_peft_adapter(PeftModel: Any, base_model: Any, adapter_dir: str, trainable: bool) -> Any:
    try:
        return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)
    except TypeError:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        if trainable:
            for name, param in model.named_parameters():
                if "lora_" in name.lower() or "modules_to_save" in name.lower():
                    param.requires_grad = True
        return model


def main() -> None:
    args = parse_args()
    os.environ["NO_POSITION_IDS"] = "1"
    torch, F, PeftModel, get_model_processor, DataLoader, Dataset, get_scheduler = _lazy_import_training_deps()
    torch.manual_seed(args.seed)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    out_dir = Path(args.output_dir)
    setup_run_logs(out_dir)

    rows = load_preference_rows_many(args.preference_file, args.max_samples)
    train_rows, eval_rows = split_train_eval(rows, args.val_ratio, args.seed)
    print(f"[data] preferences train={len(train_rows)} eval={len(eval_rows)} source={args.preference_file}")

    print("[1/5] Loading policy model from base + init LoRA...")
    model, processor_or_tokenizer = get_model_processor(
        args.model,
        model_type=args.model_type,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.torch_dtype],
        trust_remote_code=True,
    )
    model = load_peft_adapter(PeftModel, model, args.init_lora_dir, trainable=True)
    tokenizer = get_tokenizer(processor_or_tokenizer)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id

    reference_model = None
    if args.reference_mode == "frozen_init":
        ref_lora = args.reference_lora_dir or args.init_lora_dir
        print(f"[ref] Loading frozen reference model from {ref_lora}")
        reference_model, _ = get_model_processor(
            args.model,
            model_type=args.model_type,
            device_map=args.device_map,
            torch_dtype=dtype_map[args.torch_dtype],
            trust_remote_code=True,
        )
        reference_model = load_peft_adapter(PeftModel, reference_model, ref_lora, trainable=False)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False
    else:
        print("[ref] reference_free mode enabled")

    forbidden = [x.strip().lower() for x in args.forbidden_trainable_keywords.split(",") if x.strip()]
    frozen = freeze_forbidden_params(model, forbidden)
    if frozen:
        print(f"[audit] Auto-froze {frozen} forbidden params")
    summary = summarize_trainable_params(model, out_dir, forbidden)
    print(f"[audit] trainable={summary['total_trainable_params']:,} forbidden={summary['forbidden_trainable_count']}")
    if summary["forbidden_trainable_count"]:
        raise RuntimeError(f"Forbidden trainable params found. Check {out_dir / 'dpo_freeze_summary.json'}")
    if summary["total_trainable_params"] == 0:
        raise RuntimeError("No trainable LoRA parameters found. Check --init-lora-dir and PEFT loading.")

    system_prompt = ""
    if args.prompt_format == "chat_template":
        system_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
        print(f"[prompt] chat_template system_prompt={args.system_prompt} chars={len(system_prompt)}")
    else:
        print("[prompt] json_instruction compatibility mode")

    def model_device() -> Any:
        return next(model.parameters()).device

    class PreferenceDataset(Dataset):
        def __init__(self, items: list[PreferenceRow]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def _prompt_ids(self, row: PreferenceRow) -> list[int]:
            if args.prompt_format == "chat_template" and row.history and row.current_query:
                messages = format_memory_messages(row, system_prompt)
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return normalize_token_ids(ids)
            return tokenizer.encode(row.prompt, add_special_tokens=False)

        def _encode_pair(self, row: PreferenceRow, response: str) -> dict[str, list[int]]:
            prompt_ids = self._prompt_ids(row)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            if eos_token_id is not None:
                response_ids = response_ids + [eos_token_id]
            input_ids = (prompt_ids + response_ids)[-args.max_length:]
            prompt_len = max(0, len(input_ids) - len(response_ids))
            response_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
            return {"input_ids": input_ids, "response_mask": response_mask}

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.items[idx]
            return {
                "chosen": self._encode_pair(row, row.chosen),
                "rejected": self._encode_pair(row, row.rejected),
            }

    def collate(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        def pad_side(key: str) -> dict[str, Any]:
            max_len = max(len(item[key]["input_ids"]) for item in items)
            input_ids, attention_mask, response_mask = [], [], []
            for item in items:
                ids = item[key]["input_ids"]
                mask = item[key]["response_mask"]
                pad = max_len - len(ids)
                input_ids.append(ids + [pad_token_id] * pad)
                attention_mask.append([1] * len(ids) + [0] * pad)
                response_mask.append(mask + [0] * pad)
            device = model_device()
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
                "response_mask": torch.tensor(response_mask, dtype=torch.float32, device=device),
            }

        return {"chosen": pad_side("chosen"), "rejected": pad_side("rejected")}

    def sequence_logps(batch: dict[str, Any], model_obj: Any) -> tuple[Any, Any]:
        outputs = model_obj(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]
        labels = batch["input_ids"][:, 1:]
        mask = batch["response_mask"][:, 1:] * batch["attention_mask"][:, 1:]
        token_logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        seq_logps = (token_logps * mask).sum(dim=-1)
        token_counts = mask.sum(dim=-1).clamp_min(1.0)
        return seq_logps, token_counts

    def dpo_step(batch: dict[str, dict[str, Any]], train: bool) -> dict[str, float]:
        policy_chosen, chosen_tokens = sequence_logps(batch["chosen"], model)
        policy_rejected, _ = sequence_logps(batch["rejected"], model)
        policy_logratio = policy_chosen - policy_rejected

        if reference_model is not None:
            with torch.no_grad():
                ref_chosen, _ = sequence_logps(batch["chosen"], reference_model)
                ref_rejected, _ = sequence_logps(batch["rejected"], reference_model)
                reference_logratio = ref_chosen - ref_rejected
        else:
            reference_logratio = torch.zeros_like(policy_logratio)

        logits = args.beta * (policy_logratio - reference_logratio)
        dpo_loss = -F.logsigmoid(logits).mean()
        chosen_nll = -(policy_chosen / chosen_tokens).mean()
        loss = dpo_loss + args.sft_loss_weight * chosen_nll
        if train:
            loss.backward()
        return {
            "loss": float(loss.detach().cpu()),
            "dpo_loss": float(dpo_loss.detach().cpu()),
            "chosen_nll": float(chosen_nll.detach().cpu()),
            "preference_acc": float((logits > 0).float().mean().detach().cpu()),
            "logit_margin": float(logits.mean().detach().cpu()),
        }

    train_loader = DataLoader(
        PreferenceDataset(train_rows),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        PreferenceDataset(eval_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    ) if eval_rows else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = max(1, math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum)))
    scheduler = get_scheduler(optimizer, num_warmup_steps=max(1, int(total_steps * 0.03)), num_training_steps=total_steps)

    metrics_path = out_dir / "dpo_metrics.jsonl"

    def log_metrics(record: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def evaluate(step: int) -> None:
        if eval_loader is None:
            return
        model.eval()
        totals: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for batch in eval_loader:
                metrics = dpo_step(batch, train=False)
                n += 1
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + v
        record = {"step": step, "split": "eval", **{f"eval_{k}": v / max(n, 1) for k, v in totals.items()}}
        print("[eval] " + " | ".join(f"{k}={v:.6f}" for k, v in record.items() if isinstance(v, float)))
        log_metrics(record)
        model.train()

    print("[2/5] Start DPO-style preference training...")
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            metrics = dpo_step(batch, train=True)
            if micro_step % args.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.logging_steps == 0:
                    record = {
                        "step": global_step,
                        "epoch": epoch + micro_step / max(len(train_loader), 1),
                        "split": "train",
                        "learning_rate": scheduler.get_last_lr()[0],
                        **metrics,
                    }
                    print("[train] " + " | ".join(f"{k}={v:.6f}" for k, v in record.items() if isinstance(v, float)))
                    log_metrics(record)
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    evaluate(global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(str(ckpt))
                    print(f"[save] checkpoint -> {ckpt}")

        if micro_step % args.grad_accum != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        evaluate(global_step)

    print("[5/5] Saving final LoRA adapter...")
    model.save_pretrained(str(out_dir))
    if hasattr(processor_or_tokenizer, "save_pretrained"):
        processor_or_tokenizer.save_pretrained(str(out_dir / "processor"))
    config = vars(args)
    config["effective_train_steps"] = global_step
    (out_dir / "dpo_run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training complete. DPO LoRA saved to: {out_dir}")


if __name__ == "__main__":
    main()
