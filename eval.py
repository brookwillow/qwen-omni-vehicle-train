#!/usr/bin/env python3
"""Unified evaluation script for Qwen2.5-Omni ReAct tool-calling.

Two modes:
  batch   — Evaluate on data/eval/*_test.json (tool + args matching)
  single  — Single prompt inference for manual testing

Usage:
    # Batch eval on structured test sets
    python eval.py batch --model-dir models/Qwen2.5-Omni-3B --lora-dir output/lora

    # Single prompt
    python eval.py single --model-dir models/Qwen2.5-Omni-3B --prompt "打开主驾车窗"
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Suppress the per-sample "System prompt modified" warning from Qwen2.5-Omni;
# it fires once per inference call when using a custom system prompt and is expected.
logging.getLogger().addFilter(
    type("_IgnoreQwenSPWarn", (logging.Filter,), {
        "filter": lambda self, r: "System prompt modified" not in r.getMessage()
    })()
)

ACTION_RE = re.compile(r"Action:\s*([A-Za-z0-9_]+)")
ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{[\s\S]*\})")

# Resolve default SP path relative to script location (works from any cwd).
_PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SP_PATH = str(_PROJECT_DIR / "data" / "system-prompt.txt")


# ── Model loading ────────────────────────────────────────────

def load_model(model_dir: str, lora_dir: str):
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto",
    )
    if lora_dir:
        # Workaround: PEFT/accelerate bug where _no_split_modules is a set-of-sets
        # causing "unhashable type: 'set'" in get_balanced_memory.
        if hasattr(model, "_no_split_modules"):
            raw = model._no_split_modules
            if isinstance(raw, (set, frozenset)):
                flat: list = []
                for item in raw:
                    if isinstance(item, (set, frozenset, list, tuple)):
                        flat.extend(item)
                    else:
                        flat.append(item)
                model._no_split_modules = flat
        model = PeftModel.from_pretrained(model, lora_dir)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    return model, processor


def generate_text(
    model, processor, system_prompt: str, user_query: str,
    max_new_tokens: int, temperature: float = 0.0,
    audio_path: Optional[str] = None,
) -> str:
    if audio_path:
        user_content = [{"type": "audio", "audio": audio_path}]
    else:
        user_content = [{"type": "text", "text": user_query}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(model.device)
    if getattr(model, "dtype", None) is not None:
        inputs = inputs.to(model.dtype)
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            return_audio=False,
        )
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    gen_ids = out_ids[:, input_len:] if input_len > 0 else out_ids
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return (decoded[0] if decoded else "").strip()


# ── Parsing helpers ──────────────────────────────────────────

def parse_action(pred: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """Returns (tool_name, args, pred_type).

    pred_type: 'Action' | 'Clarify' | 'Reject' | 'ParseFail'
    """
    s = (pred or "").strip()
    if s.startswith("Reject"):
        return None, None, "Reject"
    if s.startswith("Clarify"):
        return None, None, "Clarify"
    m_tool = ACTION_RE.search(s)
    if not m_tool:
        return None, None, "ParseFail"
    tool = m_tool.group(1).strip()
    m_args = ACTION_INPUT_RE.search(s)
    if not m_args:
        return tool, {}, "Action"
    try:
        args = json.loads(m_args.group(1))
        return tool, args if isinstance(args, dict) else {}, "Action"
    except Exception:
        return tool, {}, "Action"


def normalize_args(d: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(d, ensure_ascii=False, sort_keys=True))


# ── Metrics ──────────────────────────────────────────────────

@dataclass
class Metric:
    total: int = 0
    tool_hit: int = 0
    args_em: int = 0
    type_correct: int = 0
    reject_pred: int = 0
    reject_hit: int = 0
    clarify_pred: int = 0
    clarify_hit: int = 0
    parse_fail: int = 0

    def __iadd__(self, other: "Metric") -> "Metric":
        self.total += other.total
        self.tool_hit += other.tool_hit
        self.args_em += other.args_em
        self.type_correct += other.type_correct
        self.reject_pred += other.reject_pred
        self.reject_hit += other.reject_hit
        self.clarify_pred += other.clarify_pred
        self.clarify_hit += other.clarify_hit
        self.parse_fail += other.parse_fail
        return self


def fmt_rate(n: int, d: int) -> str:
    return f"{(n / d * 100):.2f}%" if d else "0.00%"


def _metric_to_dict(m: Metric) -> Dict[str, Any]:
    d: Dict[str, Any] = {"total": m.total}
    if m.total == 0:
        return d
    d["type_acc"] = round(m.type_correct / m.total, 4)
    d["tool_acc"] = round(m.tool_hit / m.total, 4)
    d["args_em"] = round(m.args_em / m.total, 4)
    d["type_correct"] = m.type_correct
    d["tool_hit"] = m.tool_hit
    d["args_em_count"] = m.args_em
    d["reject_pred"] = m.reject_pred
    d["reject_hit"] = m.reject_hit
    d["clarify_pred"] = m.clarify_pred
    d["clarify_hit"] = m.clarify_hit
    d["parse_fail"] = m.parse_fail
    return d


# ── Mode: batch ──────────────────────────────────────────────


def get_expected_type(row: dict) -> str:
    """Determine expected response type from a test sample.

    Supports an explicit ``expected_type`` field; otherwise infers from
    ``expected_tool_calls``: non-empty → Action, empty → Reject.
    """
    if "expected_type" in row:
        return row["expected_type"]
    if row.get("expected_tool_calls"):
        return "Action"
    return "Reject"


def eval_file(
    model, processor, system_prompt: str, file_path: str,
    max_new_tokens: int, temperature: float,
    max_per_file: int, show_errors: int,
    by_difficulty: Optional[Dict[str, Metric]] = None,
    by_category: Optional[Dict[str, Metric]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
) -> Metric:
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if max_per_file > 0:
        data = data[:max_per_file]
    metric = Metric()
    bad = 0
    n_total = len(data)

    eval_dir = Path(file_path).parent

    for idx, row in enumerate(data):
        query = row.get("query", "")
        gt_calls = row.get("expected_tool_calls", []) or []
        gt_tool = gt_calls[0].get("name") if gt_calls else None
        gt_args = gt_calls[0].get("arguments", {}) if gt_calls else {}
        expected_type = get_expected_type(row)
        difficulty = row.get("difficulty", "unknown")
        category = row.get("category", "unknown")

        # Use audio file when available
        audio_path = None
        raw_audio = row.get("query_audio", "")
        if raw_audio:
            ap = eval_dir / raw_audio
            if ap.exists():
                audio_path = str(ap)

        src = f"[audio]" if audio_path else f"[text]"
        q_display = (query or raw_audio or "")[:60]
        print(f"  [{idx+1}/{n_total}] {src} {q_display}", end=" ... ", flush=True)

        pred = generate_text(model, processor, system_prompt, query, max_new_tokens, temperature, audio_path=audio_path)
        pred_tool, pred_args, pred_type = parse_action(pred)

        # Per-sample inline result
        type_ok = "✓" if pred_type == expected_type else "✗"
        print(f"{type_ok} {pred_type}({pred_tool or '-'})", flush=True)

        # Build per-sample metric
        sm = Metric(total=1)
        sm.reject_pred = int(pred_type == "Reject")
        sm.clarify_pred = int(pred_type == "Clarify")
        sm.parse_fail = int(pred_type == "ParseFail")
        sm.type_correct = int(pred_type == expected_type)

        if expected_type == "Reject":
            if pred_type == "Reject":
                sm.reject_hit = 1
                sm.tool_hit = 1
                sm.args_em = 1
        elif expected_type == "Clarify":
            if pred_type == "Clarify":
                sm.clarify_hit = 1
                sm.tool_hit = 1
                sm.args_em = 1
        else:  # Action
            if pred_type == "Action" and pred_tool == gt_tool:
                sm.tool_hit = 1
                if normalize_args(pred_args or {}) == normalize_args(gt_args or {}):
                    sm.args_em = 1

        # Collect & print errors
        err_type = None
        if sm.tool_hit == 0:
            if pred_type != expected_type:
                err_type = "type-err"
            elif expected_type == "Action" and pred_tool != gt_tool:
                err_type = "tool-err"
            else:
                err_type = "miss"
        elif sm.args_em == 0:
            err_type = "args-err"

        if err_type:
            if errors is not None:
                errors.append({
                    "id": row.get("id", ""),
                    "file": Path(file_path).name,
                    "err_type": err_type,
                    "query": query,
                    "expected_type": expected_type,
                    "gt_tool": gt_tool,
                    "gt_args": gt_args,
                    "pred_type": pred_type,
                    "pred_tool": pred_tool,
                    "pred_args": pred_args,
                    "pred_raw": pred[:300],
                    "difficulty": difficulty,
                    "category": category,
                })
            if bad < show_errors:
                if err_type == "type-err":
                    print(f"  [type-err] q={query}")
                    print(f"        expected={expected_type}({gt_tool or ''}), pred={pred_type}")
                    print(f"        raw={pred[:200].replace(chr(10), ' | ')}")
                elif err_type == "tool-err":
                    print(f"  [tool-err] q={query}")
                    print(f"        gt={gt_tool} {gt_args}")
                    print(f"        pred={pred_tool} {pred_args}")
                elif err_type == "args-err":
                    print(f"  [args-err] q={query}")
                    print(f"        gt={gt_args}")
                    print(f"        pred={pred_args}")
                bad += 1

        metric += sm
        if by_difficulty is not None:
            by_difficulty.setdefault(difficulty, Metric())
            by_difficulty[difficulty] += sm
        if by_category is not None:
            by_category.setdefault(category, Metric())
            by_category[category] += sm

    return metric


def _fmt_summary(m: Metric) -> str:
    parts = [
        f"n={m.total}",
        f"type_acc={fmt_rate(m.type_correct, m.total)}",
        f"tool_acc={fmt_rate(m.tool_hit, m.total)}",
        f"args_em={fmt_rate(m.args_em, m.total)}",
    ]
    if m.reject_pred or m.reject_hit:
        parts.append(f"reject={m.reject_hit}/{m.reject_pred}")
    if m.clarify_pred or m.clarify_hit:
        parts.append(f"clarify={m.clarify_hit}/{m.clarify_pred}")
    if m.parse_fail:
        parts.append(f"parse_fail={m.parse_fail}")
    return " | ".join(parts)


def run_batch(args, model, processor, system_prompt: str) -> None:
    files = sorted(glob.glob(str(Path(args.eval_dir) / args.pattern)))
    if not files:
        raise SystemExit(f"No eval files matched: {args.eval_dir}/{args.pattern}")

    total = Metric()
    by_difficulty: Dict[str, Metric] = {}
    by_category: Dict[str, Metric] = {}
    per_file: Dict[str, Metric] = {}
    errors: List[Dict[str, Any]] = []

    print(f"[eval] files={len(files)} pattern={args.pattern} max_per_file={args.max_per_file}")
    for f in files:
        fname = Path(f).name
        print(f"\n== {fname} ==")
        m = eval_file(
            model, processor, system_prompt, f,
            args.max_new_tokens, args.temperature,
            args.max_per_file, args.show_errors,
            by_difficulty, by_category, errors,
        )
        total += m
        per_file[fname] = m
        print(f"  {_fmt_summary(m)}")

    print("\n" + "=" * 60)
    print(f"OVERALL  {_fmt_summary(total)}")

    # ── Difficulty breakdown ──
    print("\n---- By Difficulty ----")
    for d in ["easy", "medium", "hard"]:
        dm = by_difficulty.get(d)
        if not dm:
            continue
        print(
            f"  {d:8s} | n={dm.total:4d}"
            f" | type_acc={fmt_rate(dm.type_correct, dm.total)}"
            f" | tool_acc={fmt_rate(dm.tool_hit, dm.total)}"
            f" | args_em={fmt_rate(dm.args_em, dm.total)}"
        )

    # ── Category breakdown (worst tool_acc first) ──
    print("\n---- By Category (worst tool_acc first, top 10) ----")
    cat_sorted = sorted(
        by_category.items(),
        key=lambda kv: kv[1].tool_hit / max(kv[1].total, 1),
    )
    for name, cm in cat_sorted[:10]:
        print(
            f"  {name:20s} | n={cm.total:3d}"
            f" | tool_acc={fmt_rate(cm.tool_hit, cm.total)}"
            f" | args_em={fmt_rate(cm.args_em, cm.total)}"
        )

    # ── Write JSON report ──
    report_path = getattr(args, "report", "") or ""
    if not report_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"eval_report_{ts}.json"

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_dir": args.model_dir,
        "lora_dir": args.lora_dir or None,
        "eval_dir": args.eval_dir,
        "pattern": args.pattern,
        "max_per_file": args.max_per_file,
        "temperature": args.temperature,
        "overall": _metric_to_dict(total),
        "by_file": {k: _metric_to_dict(v) for k, v in per_file.items()},
        "by_difficulty": {k: _metric_to_dict(v) for k, v in by_difficulty.items()},
        "by_category": {
            k: _metric_to_dict(v)
            for k, v in sorted(by_category.items(), key=lambda kv: kv[1].tool_hit / max(kv[1].total, 1))
        },
        "errors": errors,
    }
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n[report] saved to {report_path} ({len(errors)} errors logged)")


# ── Mode: single ─────────────────────────────────────────────

def run_single(args, model, processor, system_prompt: str) -> None:
    audio_path = getattr(args, "audio", None) or None
    if not audio_path and not args.prompt:
        raise SystemExit("error: --prompt is required when --audio is not provided.")
    if audio_path:
        print(f"[audio] {audio_path}")
    pred = generate_text(
        model, processor, system_prompt, args.prompt, args.max_new_tokens,
        audio_path=audio_path,
    )
    tool, tool_args, pred_type = parse_action(pred)
    print(f"[type] {pred_type}")
    if pred_type == "Action":
        print(f"[tool] {tool}")
        print(f"[args] {json.dumps(tool_args, ensure_ascii=False)}")
    print(f"[raw]  {pred}")


# ── CLI ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni ReAct tool-calling.")

    # Common arguments shared by both subcommands (defined here AND in subparsers
    # via `parents` so users can place them either before or after the subcommand).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-dir", required=True)
    common.add_argument("--lora-dir", default="", help="Optional LoRA adapter directory.")
    common.add_argument("--system-prompt-file", default=DEFAULT_SP_PATH, help="System prompt text file.")
    common.add_argument("--max-new-tokens", type=int, default=128)

    sub = p.add_subparsers(dest="mode", required=True)

    # batch
    b = sub.add_parser("batch", parents=[common], help="Evaluate on data/eval test JSON files.")
    b.add_argument("--eval-dir", default="data/eval")
    b.add_argument("--pattern", default="*_test.json")
    b.add_argument("--temperature", type=float, default=0.0)
    b.add_argument("--max-per-file", type=int, default=0, help="0 = all samples")
    b.add_argument("--show-errors", type=int, default=3)
    b.add_argument("--report", default="", help="Output JSON report path (default: eval_report_<ts>.json).")

    # single
    s = sub.add_parser("single", parents=[common], help="Single prompt inference.")
    s.add_argument("--prompt", default="", help="Text prompt (required unless --audio is provided).")
    s.add_argument("--audio", default="", help="Optional audio file path (uses audio instead of text).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sp_path = Path(args.system_prompt_file)
    if not sp_path.exists():
        raise SystemExit(f"System prompt file not found: {sp_path}")
    system_prompt = sp_path.read_text(encoding="utf-8").strip()
    print(f"[sp] {sp_path} chars={len(system_prompt)}")

    model, processor = load_model(args.model_dir, args.lora_dir)
    print(f"[model] base={args.model_dir}")
    print(f"[lora] {'none' if not args.lora_dir else args.lora_dir}")

    if args.mode == "batch":
        run_batch(args, model, processor, system_prompt)
    elif args.mode == "single":
        run_single(args, model, processor, system_prompt)


if __name__ == "__main__":
    main()
