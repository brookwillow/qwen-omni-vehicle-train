#!/usr/bin/env python3
"""
Build final training JSONL by merging split data files + injecting system prompt.

Reads JSONL files from data/splits/ recursively, prepends the system prompt
from a single file, shuffles, and writes the final training dataset.

Usage:
    # Default: merge all splits
    python build_train_data.py

    # Custom selection
    python build_train_data.py \
        --splits data/splits/action.jsonl data/splits/clarify.jsonl data/splits/reject.jsonl \
        --sp-file data/system-prompt.txt \
        --output data/train_final.jsonl

    # Oversample reject to balance
    python build_train_data.py --oversample reject:2

    # Limit samples per type
    python build_train_data.py --max-per-type action:1000 clarify:500 reject:500
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build training JSONL from split files + system prompt")
    p.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory containing split JSONL files. Defaults to recursive discovery.",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=[],
        help="Explicit list of split JSONL files. If empty, uses all *.jsonl recursively in --splits-dir",
    )
    p.add_argument(
        "--sp-file",
        default="data/system-prompt.txt",
        help="System prompt text file",
    )
    p.add_argument(
        "--macro-sp",
        action="store_true",
        help="注入前对 SP 做枚举宏压缩(scripts/build_macro_sp.py), "
             "并写出 <sp-file同目录>/system-prompt-macro.txt 供推理/评估使用",
    )
    p.add_argument(
        "--output",
        default="data/train_final.jsonl",
        help="Output training JSONL",
    )
    p.add_argument(
        "--oversample",
        nargs="*",
        default=[],
        help="Oversample specific types, e.g. reject:2 clarify:1.5",
    )
    p.add_argument(
        "--max-per-type",
        nargs="*",
        default=[],
        help="Max samples per type, e.g. action:1000 reject:500",
    )
    p.add_argument(
        "--sample-weight",
        nargs="*",
        default=[],
        help=(
            "Oversample matching split files by stem/path/glob, e.g. "
            "hard_cases/*.jsonl:3 WindowControl:1.5"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.0, help="If >0, also write a val split")
    p.add_argument(
        "--validate-schema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate tool calls against tools.json schema (default: on)",
    )
    p.add_argument(
        "--tools-file",
        default="data/tools.json",
        help="Tool schema file for validation",
    )
    return p.parse_args()


def parse_kv_args(args_list: list[str]) -> dict[str, float]:
    """Parse key:value pairs like ['reject:2', 'clarify:1.5']."""
    result = {}
    for item in args_list:
        if ":" in item:
            k, v = item.split(":", 1)
            result[k] = float(v)
    return result


def load_split(path: str) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def match_weight_selector(path: Path, selector: str) -> bool:
    """Return True when selector matches a split file by stem, path suffix, or glob."""
    normalized = path.as_posix()
    selector = selector.strip()
    if not selector:
        return False
    if selector == path.stem:
        return True
    if normalized.endswith(selector):
        return True
    return path.match(selector) or Path(normalized).match(selector)


def sample_weight_for_path(path: Path, sample_weights: dict[str, float]) -> float:
    downsample_factor = 1.0
    oversample_factor = 1.0
    for selector, weight in sample_weights.items():
        if match_weight_selector(path, selector):
            if weight < 1.0:
                downsample_factor = min(downsample_factor, weight)
            else:
                oversample_factor = max(oversample_factor, weight)
    if downsample_factor < 1.0:
        return downsample_factor
    return oversample_factor


def expand_samples(samples: list[dict], factor: float, rng: random.Random) -> list[dict]:
    """Apply deterministic fractional sampling.

    ``factor > 1`` oversamples. ``0 < factor < 1`` downsamples without
    replacement, which is useful for overrepresented boundary classes such as
    current-turn noise.
    """
    if factor <= 0:
        return []
    if factor < 1.0:
        keep = int(len(samples) * factor)
        if len(samples) and keep == 0:
            keep = 1
        shuffled = list(samples)
        rng.shuffle(shuffled)
        return shuffled[:keep]
    if factor == 1.0:
        return samples
    int_factor = int(factor)
    frac = factor - int_factor
    expanded = samples * int_factor
    if frac > 0:
        extra = int(len(samples) * frac)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        expanded += shuffled[:extra]
    return expanded


def load_weighted_splits(
    split_files: list[Path],
    oversample: dict[str, float],
    max_per_type: dict[str, float],
    sample_weights: dict[str, float],
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    all_samples = []
    counts = {}
    for fpath in split_files:
        type_name = fpath.stem
        samples = load_split(str(fpath))

        if type_name in max_per_type:
            limit = int(max_per_type[type_name])
            if len(samples) > limit:
                rng.shuffle(samples)
                samples = samples[:limit]

        oversample_factor = oversample.get(type_name, 1.0)
        path_factor = sample_weight_for_path(fpath, sample_weights)
        factor = min(oversample_factor, path_factor) if path_factor < 1.0 else max(oversample_factor, path_factor)
        samples = expand_samples(samples, factor, rng)

        counts[str(fpath)] = len(samples)
        all_samples.extend(samples)
    return all_samples, counts


def inject_sp(sample: dict, sp: str) -> dict:
    """Prepend system prompt to messages."""
    msgs = [{"role": "system", "content": sp}] + sample["messages"]
    return {"messages": msgs}


def is_tool_call_content(content: str) -> bool:
    content = content.strip()
    if content.startswith("Action:"):
        return True
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return False
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        if data["name"] == "NoiseDoNotAct":
            return True
        return "arguments" in data
    if isinstance(data, list):
        return all(
            isinstance(item, dict) and isinstance(item.get("name"), str) and "arguments" in item
            for item in data
        )
    return False


NUMERIC_VALUE_RE = re.compile(r"^\d+(\.\d+)?%?$")


def load_tool_schema(tools_path: str) -> dict:
    """Load tool schema from tools.json and return {name: {props, required}}."""
    with open(tools_path, encoding="utf-8") as f:
        tools = json.load(f)
    schema = {}
    for t in tools:
        fn = t.get("function", t)
        name = fn["name"]
        params = fn.get("parameters") or fn.get("inputSchema") or {}
        props = params.get("properties", {})
        required = params.get("required", [])
        schema[name] = {"props": props, "required": required}
    return schema


def _parse_tool_calls(content: str):
    """Parse one or more tool calls from assistant content."""
    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        args = data.get("arguments")
        return [(data["name"], args if isinstance(args, dict) else None)]
    if isinstance(data, list):
        calls = []
        for item in data:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or "arguments" not in item
            ):
                return []
            args = item.get("arguments")
            calls.append((item["name"], args if isinstance(args, dict) else None))
        return calls
    m = re.match(r"Action:\s*(\S+)\s*\nAction Input:\s*(\{.*\})", content, re.DOTALL)
    if not m:
        return []
    tool_name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return [(tool_name, None)]
    return [(tool_name, args)]


def validate_sample_schema(sample: dict, schema: dict) -> list[str]:
    """Validate all tool calls in a sample against the tool schema.

    Returns a list of human-readable error strings (empty = valid).
    """
    errors: list[str] = []
    for msg in sample.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "").strip()
        if not (content.startswith(("{", "[")) or content.startswith("Action:")):
            continue
        tool_calls = _parse_tool_calls(content)
        if not tool_calls:
            if content.startswith("["):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data = None
                is_toolish_array = isinstance(data, list) and any(
                    isinstance(item, dict)
                    and (item.get("name") in schema or "arguments" in item)
                    for item in data
                )
                if not is_toolish_array:
                    continue
            errors.append("failed to parse tool call content")
            continue
        for tool_name, args in tool_calls:
            if tool_name not in schema:
                errors.append(f"unknown tool {tool_name!r}")
                continue
            if tool_name == "NoiseDoNotAct":
                continue
            if args is None:
                errors.append(f"{tool_name}: failed to parse arguments")
                continue
            s = schema[tool_name]
            for req in s["required"]:
                if req not in args:
                    errors.append(f"{tool_name}: missing required field {req!r}")
            for param, val in args.items():
                if param not in s["props"]:
                    errors.append(f"{tool_name}: unknown field {param!r}")
                    continue
                prop = s["props"][param]
                if "enum" not in prop:
                    continue
                if val in prop["enum"]:
                    continue
                # Allow numeric/percentage strings for value-like fields
                if param == "value" and NUMERIC_VALUE_RE.match(str(val)):
                    continue
                desc = prop.get("description", "")
                if ("numeric" in desc or "percentage" in desc) and NUMERIC_VALUE_RE.match(str(val)):
                    continue
                errors.append(f"{tool_name}.{param}: invalid enum {val!r}")
    return errors


def validate_all_samples(
    samples: list[dict], schema: dict, label: str
) -> tuple[list[dict], int]:
    """Validate samples against schema, print warnings, return (clean, n_bad)."""
    clean = []
    n_bad = 0
    for i, sample in enumerate(samples):
        errs = validate_sample_schema(sample, schema)
        if errs:
            n_bad += 1
            for err in errs:
                print(f"[schema-warn] {label} sample {i}: {err}", file=sys.stderr)
        else:
            clean.append(sample)
    return clean, n_bad


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Load system prompt
    sp_path = Path(args.sp_file)
    if not sp_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {sp_path}")
    sp = sp_path.read_text(encoding="utf-8").strip()
    print(f"[sp] loaded from {sp_path} ({len(sp)} chars)")

    if args.macro_sp:
        from scripts.build_macro_sp import build_macro_sp

        macro_text, macros = build_macro_sp(sp + "\n")
        macro_path = sp_path.with_name("system-prompt-macro.txt")
        macro_path.write_text(macro_text, encoding="utf-8")
        sp = macro_text.strip()
        print(f"[sp] macro-compressed: {len(sp)} chars, {len(macros)} macros → {macro_path}")
        print(f"[sp] 训练/推理/评估请统一使用 --system-prompt-file {macro_path}")

    # Discover split files
    if args.splits:
        split_files = [Path(p) for p in args.splits]
    else:
        splits_dir = Path(args.splits_dir)
        if not splits_dir.exists():
            raise FileNotFoundError(
                f"Splits directory not found: {splits_dir}\n"
                f"Run: python split_data_by_type.py"
            )
        split_files = sorted(splits_dir.rglob("*.jsonl"))

    if not split_files:
        raise FileNotFoundError("No split files found")

    # Parse oversampling and max-per-type
    oversample = parse_kv_args(args.oversample)
    max_per_type = parse_kv_args(args.max_per_type)
    sample_weights = parse_kv_args(args.sample_weight)

    # Load and process splits
    all_samples, counts = load_weighted_splits(split_files, oversample, max_per_type, sample_weights, rng)
    for fpath in split_files:
        print(f"  {fpath.stem}: {counts[str(fpath)]} samples from {fpath}")

    # Schema validation
    if args.validate_schema:
        tools_path = Path(args.tools_file)
        if not tools_path.exists():
            print(f"[warn] tools file not found: {tools_path}, skipping schema validation")
        else:
            schema = load_tool_schema(str(tools_path))
            all_samples, n_bad = validate_all_samples(all_samples, schema, "merged")
            if n_bad:
                print(f"[schema] Removed {n_bad} samples with schema violations")
            print(f"[schema] {len(all_samples)} samples passed validation")

    # Inject SP and shuffle
    final = [inject_sp(s, sp) for s in all_samples]
    rng.shuffle(final)

    # Optional val split
    if args.val_ratio > 0:
        val_size = int(len(final) * args.val_ratio)
        val_set = final[:val_size]
        final = final[val_size:]
        val_path = Path(args.output).with_suffix(".val.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_set:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n[val] {len(val_set)} samples → {val_path}")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[out] {len(final)} samples → {out_path}")

    # Print distribution (classify by last assistant turn, not all turns)
    dist = {"Action": 0, "MultiAction": 0, "Noise": 0, "TTS": 0, "Reject": 0}
    for s in final:
        label = None
        for m in s["messages"]:
            if m["role"] == "assistant":
                c = m["content"].strip()
                if c.startswith("["):
                    label = "MultiAction"
                elif c.startswith("{"):
                    try:
                        data = json.loads(c)
                    except json.JSONDecodeError:
                        data = None
                    if isinstance(data, dict) and data.get("name") == "NoiseDoNotAct":
                        label = "Noise"
                    elif is_tool_call_content(c):
                        label = "Action"
                    else:
                        label = "TTS"
                elif is_tool_call_content(c):
                    label = "Action"
                elif c.startswith("Reject"):
                    label = "Reject"
                elif c:
                    label = "TTS"
        if label:
            dist[label] += 1
    print(f"\n[stats] Distribution ({len(final)} samples):")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/len(final)*100:.1f}%)")


if __name__ == "__main__":
    main()
