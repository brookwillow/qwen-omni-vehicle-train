#!/usr/bin/env python3
"""
Build final training JSONL by merging split data files + injecting system prompt.

Reads type-specific JSONL files from data/splits/, prepends the system prompt
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
        help="Directory containing split JSONL files",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=[],
        help="Explicit list of split JSONL files. If empty, uses all *.jsonl in --splits-dir",
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


def inject_sp(sample: dict, sp: str) -> dict:
    """Prepend system prompt to messages."""
    msgs = [{"role": "system", "content": sp}] + sample["messages"]
    return {"messages": msgs}


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
        split_files = sorted(splits_dir.glob("*.jsonl"))

    if not split_files:
        raise FileNotFoundError("No split files found")

    # Parse oversampling and max-per-type
    oversample = parse_kv_args(args.oversample)
    max_per_type = parse_kv_args(args.max_per_type)

    # Load and process splits
    all_samples = []
    for fpath in split_files:
        type_name = fpath.stem  # e.g. "action", "reject"
        samples = load_split(str(fpath))

        # Apply max-per-type limit
        if type_name in max_per_type:
            limit = int(max_per_type[type_name])
            if len(samples) > limit:
                rng.shuffle(samples)
                samples = samples[:limit]

        # Apply oversampling
        factor = oversample.get(type_name, 1.0)
        if factor > 1.0:
            int_factor = int(factor)
            frac = factor - int_factor
            expanded = samples * int_factor
            if frac > 0:
                extra = int(len(samples) * frac)
                rng.shuffle(samples)
                expanded += samples[:extra]
            samples = expanded

        print(f"  {type_name}: {len(samples)} samples from {fpath}")
        all_samples.extend(samples)

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
    dist = {"Action": 0, "Clarify": 0, "Reject": 0, "FinalAnswer": 0}
    for s in final:
        # Find the last assistant message that is Action/Clarify/Reject
        label = None
        for m in s["messages"]:
            if m["role"] == "assistant":
                c = m["content"].strip()
                if c.startswith("Action:"):
                    label = "Action"
                elif c.startswith("Clarify:"):
                    label = "Clarify"
                elif c.startswith("Reject"):
                    label = "Reject"
                elif "Final Answer" in c:
                    label = label or "FinalAnswer"
        if label:
            dist[label] += 1
    print(f"\n[stats] Distribution ({len(final)} samples):")
    for k, v in dist.items():
        print(f"  {k}: {v} ({v/len(final)*100:.1f}%)")


if __name__ == "__main__":
    main()
