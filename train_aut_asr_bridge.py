#!/usr/bin/env python3
"""Train a small AUT-to-Whisper ASR bridge on eval audio.

This script is intentionally self-contained:
- collects ``query`` + ``query_audio`` pairs from ``data/eval/*_test.json``;
- freezes Qwen2.5-Omni and Whisper;
- hooks Qwen ``thinker.audio_tower.ln_post`` hidden states;
- trains only a lightweight bridge before the frozen Whisper decoder;
- evaluates after each epoch and writes metrics/sample predictions.

Example:
  python train_aut_asr_bridge.py \
    --model-dir /home/wangjie/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-3B \
    --whisper-dir openai/whisper-large-v3 \
    --eval-dir data/eval \
    --output-dir aut_asr_bridge_output \
    --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, TextIO

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class AsrRow:
    id: str
    query: str
    audio: str
    eval_file: str
    category: str = ""
    sub_category: str = ""
    intent: str = ""


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


class AutAsrBridge(nn.Module):
    """Small residual bridge from Qwen audio hidden states to Whisper space."""

    def __init__(
        self,
        dim: int = 1280,
        hidden_dim: int = 1280,
        dropout: float = 0.05,
        repeat_factor: int = 4,
    ) -> None:
        super().__init__()
        self.repeat_factor = max(1, repeat_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.repeat_factor > 1:
            hidden = hidden.repeat_interleave(self.repeat_factor, dim=1)
        return self.out_norm(hidden + self.net(hidden))


class HiddenDataset(Dataset):
    def __init__(self, rows: list[AsrRow], tokenizer: Any, cache_dir: Path, language: str) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.language = language
        if hasattr(self.tokenizer, "set_prefix_tokens"):
            self.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        cache_path = hidden_cache_path(self.cache_dir, row)
        hidden = torch.load(cache_path, map_location="cpu")
        tokenized = self.tokenizer(row.query, return_tensors="pt")
        labels = tokenized.input_ids[0].long()
        return {"row": row, "hidden": hidden.float(), "labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AUT hidden states -> Whisper decoder ASR bridge.")
    parser.add_argument("--model-dir", required=True, help="Qwen2.5-Omni model directory.")
    parser.add_argument("--whisper-dir", default="openai/whisper-large-v3")
    parser.add_argument("--eval-dir", default="data/eval")
    parser.add_argument("--audio-root", default="data/eval")
    parser.add_argument("--output-dir", default="aut_asr_bridge_output")
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for smoke runs; 0 means all.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--hook-layer", default="ln_post", choices=["ln_post", "avg_pooler", "full"])
    parser.add_argument("--torch-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--bridge-dtype", default="float32", choices=["float32", "float16", "bfloat16", "model"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bridge-hidden-dim", type=int, default=1280)
    parser.add_argument("--bridge-dropout", type=float, default=0.05)
    parser.add_argument("--repeat-factor", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=32, help="Generated sample count per eval.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Only build cache and run eval if checkpoint exists.")
    return parser.parse_args()


def setup_run_logs(output_dir: Path) -> TextIO:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    log_path = output_dir / "train.log"
    metrics_path.write_text("", encoding="utf-8")
    train_log = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, train_log)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.stderr, train_log)  # type: ignore[assignment]
    print(f"[log] metrics reset: {metrics_path}")
    print(f"[log] console log: {log_path}")
    return train_log


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_whisper_dir(whisper_dir: str) -> str:
    if Path(whisper_dir).exists():
        return whisper_dir
    modelscope_ids = {
        "openai/whisper-large-v3": "AI-ModelScope/whisper-large-v3",
        "openai/whisper-large-v2": "AI-ModelScope/whisper-large-v2",
        "openai/whisper-medium": "AI-ModelScope/whisper-medium",
    }
    ms_id = modelscope_ids.get(whisper_dir, whisper_dir)
    fallback_cache = Path(os.path.expanduser("~/.cache/modelscope/hub/models")) / ms_id
    if fallback_cache.is_dir() and (fallback_cache / "config.json").exists():
        return str(fallback_cache)
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"[modelscope] downloading Whisper: {ms_id}")
        return snapshot_download(ms_id)
    except Exception as exc:
        if fallback_cache.is_dir() and (fallback_cache / "config.json").exists():
            return str(fallback_cache)
        print(f"[modelscope] unavailable ({exc}); fallback to {whisper_dir}")
        return whisper_dir


def collect_eval_rows(eval_dir: Path, audio_root: Path, limit: int = 0) -> list[AsrRow]:
    rows: list[AsrRow] = []
    for eval_path in sorted(eval_dir.glob("*_test.json")):
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            query = item.get("query")
            query_audio = item.get("query_audio")
            if not isinstance(query, str) or not query.strip() or not isinstance(query_audio, str):
                continue
            audio = Path(query_audio)
            if not audio.is_absolute():
                audio = audio_root / audio
            if not audio.is_file():
                continue
            rows.append(
                AsrRow(
                    id=str(item.get("id") or audio.stem),
                    query=query.strip(),
                    audio=str(audio),
                    eval_file=str(eval_path),
                    category=str(item.get("category") or ""),
                    sub_category=str(item.get("sub_category") or ""),
                    intent=str(item.get("intent") or ""),
                )
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def split_rows(rows: list[AsrRow], val_ratio: float, seed: int) -> tuple[list[AsrRow], list[AsrRow]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    if val_ratio <= 0:
        return shuffled, []
    val_size = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_size:], shuffled[:val_size]


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def first_param_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            out[key] = value
        elif torch.is_floating_point(value):
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def build_qwen_inputs(processor: Any, audio_path: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    from qwen_omni_utils import process_mm_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an assistant."}]},
        {"role": "user", "content": [{"type": "audio", "audio": str(Path(audio_path).resolve())}]},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    return move_inputs(inputs, device=device, dtype=dtype)


def find_audio_tower(model: nn.Module) -> nn.Module:
    for path in ["thinker.audio_tower", "thinker.model.audio_tower", "audio_tower"]:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            print(f"[qwen] audio tower: model.{path}")
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Could not find Qwen audio_tower.")


def select_hook_target(audio_tower: nn.Module, hook_layer: str) -> nn.Module:
    if hook_layer == "ln_post" and hasattr(audio_tower, "ln_post"):
        print("[qwen] hook target: audio_tower.ln_post")
        return audio_tower.ln_post
    if hook_layer == "avg_pooler" and hasattr(audio_tower, "avg_pooler"):
        print("[qwen] hook target: audio_tower.avg_pooler")
        return audio_tower.avg_pooler
    print("[qwen] hook target: full audio_tower")
    return audio_tower


def capture_audio_hidden(model: Any, hook_target: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
    captured: torch.Tensor | None = None

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal captured
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        elif hasattr(output, "last_hidden_state"):
            tensor = output.last_hidden_state
        else:
            tensor = output
        if not torch.is_tensor(tensor):
            raise TypeError(f"Hook output is not tensor: {type(tensor).__name__}")
        captured = tensor.detach().float().cpu()

    handle = hook_target.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model.thinker(**{k: v for k, v in inputs.items() if k != "labels"}, output_hidden_states=False)
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("Hook did not capture hidden states.")
    if captured.ndim == 2:
        captured = captured.unsqueeze(0)
    if captured.ndim != 3:
        raise RuntimeError(f"Expected [B,T,C], got {tuple(captured.shape)}")
    return captured.squeeze(0).contiguous()


def hidden_cache_path(cache_dir: Path, row: AsrRow) -> Path:
    eval_stem = Path(row.eval_file).stem
    safe_id = row.id.replace("/", "_")
    return cache_dir / eval_stem / f"{safe_id}.pt"


def ensure_hidden_cache(
    rows: list[AsrRow],
    cache_dir: Path,
    qwen_model: Any,
    qwen_processor: Any,
    hook_target: nn.Module,
    dtype: torch.dtype,
    rebuild: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = first_param_device(qwen_model)
    missing = [row for row in rows if rebuild or not hidden_cache_path(cache_dir, row).exists()]
    print(f"[cache] rows={len(rows)} missing={len(missing)} cache_dir={cache_dir}")
    for idx, row in enumerate(missing, start=1):
        path = hidden_cache_path(cache_dir, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[cache][{idx}/{len(missing)}] {row.id} {row.audio}")
        inputs = build_qwen_inputs(qwen_processor, row.audio, device, dtype)
        hidden = capture_audio_hidden(qwen_model, hook_target, inputs)
        torch.save(hidden, path)


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("This training script currently supports --train-batch-size 1 only.")
    return items[0]


def labels_to_device(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    return labels.unsqueeze(0).to(device=device, dtype=torch.long)


def hidden_to_device(hidden: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return hidden.unsqueeze(0).to(device=device, dtype=dtype)


def char_similarity(expected: str, actual: str) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return SequenceMatcher(None, expected, actual).ratio()


def is_repetitive(text: str) -> bool:
    compact = "".join(text.split())
    if len(compact) < 8:
        return False
    chars = list(compact)
    return max(chars.count(ch) for ch in set(chars)) / len(chars) >= 0.65


def whisper_forward_loss(
    bridge: AutAsrBridge,
    whisper_model: Any,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    bridge_dtype: torch.dtype,
) -> torch.Tensor:
    from transformers.modeling_outputs import BaseModelOutput

    device = first_param_device(whisper_model)
    bridged = bridge(hidden_to_device(hidden, device, bridge_dtype))
    bridged = bridged.to(dtype=next(whisper_model.parameters()).dtype)
    encoder_outputs = BaseModelOutput(last_hidden_state=bridged)
    attention_mask = torch.ones(bridged.shape[:2], dtype=torch.long, device=device)
    outputs = whisper_model(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        labels=labels_to_device(labels, device),
        use_cache=False,
    )
    return outputs.loss


def generate_asr(
    bridge: AutAsrBridge,
    whisper_model: Any,
    tokenizer: Any,
    hidden: torch.Tensor,
    bridge_dtype: torch.dtype,
    language: str,
    max_new_tokens: int,
) -> str:
    from transformers.modeling_outputs import BaseModelOutput

    device = first_param_device(whisper_model)
    bridged = bridge(hidden_to_device(hidden, device, bridge_dtype))
    bridged = bridged.to(dtype=next(whisper_model.parameters()).dtype)
    encoder_outputs = BaseModelOutput(last_hidden_state=bridged)
    attention_mask = torch.ones(bridged.shape[:2], dtype=torch.long, device=device)
    forced_ids = tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
    generated = whisper_model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        forced_decoder_ids=forced_ids,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def log_metrics(output_dir: Path, record: dict[str, Any]) -> None:
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(
    bridge: AutAsrBridge,
    whisper_model: Any,
    tokenizer: Any,
    rows: list[AsrRow],
    cache_dir: Path,
    dtype: torch.dtype,
    bridge_dtype: torch.dtype,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    bridge.eval()
    losses: list[float] = []
    similarities: list[float] = []
    predictions: list[dict[str, Any]] = []
    sample_rows = rows[: max(0, args.eval_samples)]
    dataset = HiddenDataset(rows, tokenizer.tokenizer, cache_dir, args.language)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch, num_workers=args.num_workers)
    with torch.inference_mode():
        for batch in loader:
            loss = whisper_forward_loss(bridge, whisper_model, batch["hidden"], batch["labels"], bridge_dtype)
            losses.append(float(loss.detach().cpu()))
        for row in sample_rows:
            hidden = torch.load(hidden_cache_path(cache_dir, row), map_location="cpu").float()
            pred = generate_asr(
                bridge,
                whisper_model,
                tokenizer,
                hidden,
                bridge_dtype,
                args.language,
                args.max_new_tokens,
            )
            predictions.append(
                {
                    "id": row.id,
                    "query": row.query,
                    "asr": pred,
                    "empty": not bool(pred.strip()),
                    "repetitive": is_repetitive(pred),
                    "char_similarity": round(char_similarity(row.query, pred), 4),
                }
            )
            similarities.append(char_similarity(row.query, pred))
    avg_loss = sum(losses) / max(len(losses), 1)
    avg_sim = sum(similarities) / max(len(similarities), 1)
    empty = sum(1 for item in predictions if item["empty"])
    repetitive = sum(1 for item in predictions if item["repetitive"])
    report = {
        "epoch": epoch,
        "split": "eval",
        "eval_loss": avg_loss,
        "sample_char_similarity": avg_sim,
        "sample_empty": empty,
        "sample_repetitive": repetitive,
        "sample_count": len(predictions),
    }
    pred_path = Path(args.output_dir) / f"eval_predictions_epoch_{epoch}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"[eval] epoch={epoch} loss={avg_loss:.6f} "
        f"sample_sim={avg_sim:.4f} empty={empty}/{len(predictions)} repeat={repetitive}/{len(predictions)}"
    )
    bridge.train()
    return report


def save_checkpoint(bridge: AutAsrBridge, output_dir: Path, name: str, config: dict[str, Any]) -> None:
    ckpt_dir = output_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"bridge_state_dict": bridge.state_dict(), "config": config}, ckpt_dir / "bridge.pt")


def main() -> None:
    args = parse_args()
    if args.train_batch_size != 1:
        raise ValueError("--train-batch-size must be 1 for variable-length hidden states.")

    output_dir = Path(args.output_dir)
    setup_run_logs(output_dir)
    config = vars(args).copy()
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch_dtype(args.torch_dtype)
    bridge_dtype = dtype if args.bridge_dtype == "model" else torch_dtype(args.bridge_dtype)

    rows = collect_eval_rows(Path(args.eval_dir), Path(args.audio_root), args.limit)
    if not rows:
        raise RuntimeError("No eval audio rows found.")
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    print(f"[data] total={len(rows)} train={len(train_rows)} val={len(val_rows)} val_ratio={args.val_ratio}")
    with (output_dir / "dataset_manifest.jsonl").open("w", encoding="utf-8") as f:
        for split, split_rows_ in [("train", train_rows), ("val", val_rows)]:
            for row in split_rows_:
                payload = asdict(row)
                payload["split"] = split
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"[1/5] loading Qwen: {args.model_dir}")
    qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=args.device,
    )
    qwen_processor = Qwen2_5OmniProcessor.from_pretrained(args.model_dir)
    freeze_model(qwen_model)
    audio_tower = find_audio_tower(qwen_model)
    hook_target = select_hook_target(audio_tower, args.hook_layer)

    print(f"[2/5] preparing AUT hidden cache")
    cache_dir = output_dir / "cache"
    ensure_hidden_cache(rows, cache_dir, qwen_model, qwen_processor, hook_target, dtype, args.rebuild_cache)
    del qwen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    whisper_source = resolve_whisper_dir(args.whisper_dir)
    print(f"[3/5] loading Whisper: {whisper_source}")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_source,
        torch_dtype=dtype,
        device_map=args.device,
    )
    whisper_processor = WhisperProcessor.from_pretrained(whisper_source)
    freeze_model(whisper_model)
    if hasattr(whisper_model.config, "use_cache"):
        whisper_model.config.use_cache = False

    tokenizer = whisper_processor
    if hasattr(tokenizer.tokenizer, "set_prefix_tokens"):
        tokenizer.tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    print("[4/5] initializing bridge")
    bridge = AutAsrBridge(
        dim=whisper_model.config.d_model,
        hidden_dim=args.bridge_hidden_dim,
        dropout=args.bridge_dropout,
        repeat_factor=args.repeat_factor,
    ).to(first_param_device(whisper_model), dtype=bridge_dtype)
    trainable = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(f"[bridge] trainable_params={trainable:,} repeat_factor={args.repeat_factor}")

    if args.skip_train:
        ckpt_path = output_dir / "final" / "bridge.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            bridge.load_state_dict(ckpt["bridge_state_dict"])
            print(f"[bridge] loaded checkpoint from {ckpt_path}")
        else:
            print(f"[bridge] WARNING: --skip-train but no checkpoint found at {ckpt_path}, using random weights")

    print("[5/5] start training")
    global_step = 0
    if val_rows:
        report = evaluate(bridge, whisper_model, tokenizer, val_rows, cache_dir, dtype, bridge_dtype, args, epoch=0)
        log_metrics(output_dir, report)
    if not args.skip_train:
        train_dataset = HiddenDataset(train_rows, tokenizer.tokenizer, cache_dir, args.language)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=args.num_workers,
        )
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = max(1, math.ceil(len(train_loader) * args.epochs / max(1, args.grad_accum)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        bridge.train()
        for epoch in range(1, args.epochs + 1):
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for micro_step, batch in enumerate(train_loader, start=1):
                loss = whisper_forward_loss(bridge, whisper_model, batch["hidden"], batch["labels"], bridge_dtype)
                (loss / args.grad_accum).backward()
                running_loss += float(loss.detach().cpu())
                if micro_step % args.grad_accum == 0 or micro_step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(bridge.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % 10 == 0 or global_step == 1:
                        avg_loss = running_loss / max(micro_step, 1)
                        lr = scheduler.get_last_lr()[0]
                        record = {
                            "split": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": avg_loss,
                            "lr": lr,
                        }
                        print(f"[train] epoch={epoch} step={global_step}/{total_steps} loss={avg_loss:.6f} lr={lr:.6e}")
                        log_metrics(output_dir, record)
            save_checkpoint(bridge, output_dir, f"checkpoint-epoch-{epoch}", config)
            if val_rows:
                report = evaluate(bridge, whisper_model, tokenizer, val_rows, cache_dir, dtype, bridge_dtype, args, epoch=epoch)
                log_metrics(output_dir, report)

    save_checkpoint(bridge, output_dir, "final", config)
    print(f"[done] bridge saved to {output_dir / 'final' / 'bridge.pt'}")


if __name__ == "__main__":
    main()
