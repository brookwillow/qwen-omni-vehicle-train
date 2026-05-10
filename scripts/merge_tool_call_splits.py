#!/usr/bin/env python3
"""
Merge all by_tool/*.jsonl back into tool_call.jsonl after cleaning.

Skips _misplaced.jsonl (those should be moved to reject/noise manually).

Run:
  python3 scripts/merge_tool_call_splits.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
BY_TOOL_DIR = SPLITS_DIR / "by_tool"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(f for f in BY_TOOL_DIR.glob("*.jsonl") if not f.name.startswith("_"))

    all_records: list[dict] = []
    for f in files:
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        print(f"  {len(recs):4d}  ← by_tool/{f.name}")
        all_records.extend(recs)

    misplaced_path = BY_TOOL_DIR / "_misplaced.jsonl"
    if misplaced_path.exists():
        mp = [json.loads(l) for l in open(misplaced_path) if l.strip()]
        print(f"  {len(mp):4d}  ← by_tool/_misplaced.jsonl  [SKIPPED — move to reject manually]")

    print(f"\nTotal to write: {len(all_records)}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    out = SPLITS_DIR / "tool_call.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[write] {out}  ({len(all_records)} records)")


if __name__ == "__main__":
    main()
