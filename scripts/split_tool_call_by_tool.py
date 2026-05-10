#!/usr/bin/env python3
"""
Split tool_call.jsonl into per-tool files for easier cleaning.

Output structure:
  data/splits/by_tool/{ToolName}.jsonl   — one file per tool
  data/splits/by_tool/_misplaced.jsonl   — records that aren't tool calls (Reject/Clarify etc.)

Run:
  python3 scripts/split_tool_call_by_tool.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
BY_TOOL_DIR = SPLITS_DIR / "by_tool"
BACKUP_PATH = SPLITS_DIR / "tool_call.jsonl.bak"


def extract_tool(rec: dict) -> str | None:
    """Return the first Action tool name found in any assistant turn, or None."""
    for msg in rec.get("messages", []):
        if msg.get("role") == "assistant":
            m = re.match(r"Action:\s*(\w+)", msg.get("content", ""))
            if m:
                return m.group(1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = SPLITS_DIR / "tool_call.jsonl"
    records = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]

    by_tool: dict[str, list] = defaultdict(list)
    misplaced: list[dict] = []

    for rec in records:
        tool = extract_tool(rec)
        if tool:
            by_tool[tool].append(rec)
        else:
            misplaced.append(rec)

    counts = Counter({t: len(v) for t, v in by_tool.items()})
    print(f"Total records: {len(records)}")
    print(f"Misplaced (no Action found): {len(misplaced)}")
    print()
    print("=== Per-tool counts ===")
    for tool, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:4d}  {tool}")

    if misplaced:
        print(f"\n=== Misplaced samples ({len(misplaced)}) ===")
        for r in misplaced:
            user_q = r["messages"][0]["content"] if r["messages"] else ""
            asst = r["messages"][1]["content"] if len(r["messages"]) > 1 else ""
            print(f"  U: {user_q!r}  →  A: {asst!r}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Backup original
    shutil.copy(src, BACKUP_PATH)
    print(f"\n[backup] {BACKUP_PATH}")

    # Write by_tool/
    BY_TOOL_DIR.mkdir(exist_ok=True)
    # Clean up old files
    for f in BY_TOOL_DIR.glob("*.jsonl"):
        f.unlink()

    print("\n[write] by_tool/")
    for tool, recs in sorted(by_tool.items()):
        out = BY_TOOL_DIR / f"{tool}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(recs):4d}  → by_tool/{tool}.jsonl")

    if misplaced:
        out = BY_TOOL_DIR / "_misplaced.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in misplaced:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {len(misplaced):4d}  → by_tool/_misplaced.jsonl")

    print("\n[done]")
    print("Edit files under data/splits/by_tool/, then run merge_tool_call_splits.py to rebuild tool_call.jsonl")


if __name__ == "__main__":
    main()
