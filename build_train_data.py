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
    factor = 1.0
    for selector, weight in sample_weights.items():
        if match_weight_selector(path, selector):
            factor = max(factor, weight)
    return factor


def expand_samples(samples: list[dict], factor: float, rng: random.Random) -> list[dict]:
    """Apply deterministic fractional oversampling."""
    if factor <= 1.0:
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

        factor = max(oversample.get(type_name, 1.0), sample_weight_for_path(fpath, sample_weights))
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
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        return False
    if data["name"] == "NoiseDoNotAct":
        return True
    return "arguments" in data


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Load system prompt
    sp_path = Path(args.sp_file)
    if not sp_path.exists():
        raise FileNotFoundError(f"System prompt file not found: {sp_path}")
    sp = sp_path.read_text(encoding="utf-8").strip()
    print(f"[sp] loaded from {sp_path} ({len(sp)} chars)")

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
    dist = {"Action": 0, "TTS": 0, "Reject": 0}
    for s in final:
        label = None
        for m in s["messages"]:
            if m["role"] == "assistant":
                c = m["content"].strip()
                if is_tool_call_content(c):
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
