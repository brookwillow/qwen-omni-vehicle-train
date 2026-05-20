#!/usr/bin/env python3
"""Build direct tool-vs-wrong-tool preferences from memory tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from build_memory_preferences import (
        compact_json,
        expected_assistant_output,
        make_synthetic_rejected,
    )
except ModuleNotFoundError:
    from scripts.build_memory_preferences import (
        compact_json,
        expected_assistant_output,
        make_synthetic_rejected,
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def is_tool_json(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and bool(value.get("name"))
        and isinstance(value.get("arguments"), dict)
    )


def build_memory_contrast_preferences(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferences = []
    seen = set()
    for task in tasks:
        chosen = expected_assistant_output(task)
        rejected = make_synthetic_rejected(task)
        if not is_tool_json(chosen) or not is_tool_json(rejected):
            continue
        if compact_json(chosen) == compact_json(rejected):
            continue
        key = json.dumps(
            {
                "history": task.get("history", []),
                "current_query": task.get("current_query", ""),
                "chosen": chosen,
                "rejected": rejected,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        preferences.append(
            {
                "id": f"{task.get('id', '')}_memory_contrast",
                "task_type": "memory_contrast",
                "source_task_type": task.get("task_type", ""),
                "history": task.get("history", []),
                "current_query": task.get("current_query", ""),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": 10.0,
                "rejected_score": 2.0,
                "chosen_source": "expected_tool_json",
                "rejected_source": "synthetic_wrong_memory_tool",
                "chosen_verdict": {
                    "uses_history_correctly": True,
                    "tool_or_response_correct": True,
                    "arguments_correct": True,
                    "score": 10.0,
                    "reason": "正确根据当前轮和最近相关历史输出目标工具 JSON。",
                },
                "rejected_verdict": {
                    "uses_history_correctly": False,
                    "tool_or_response_correct": False,
                    "arguments_correct": False,
                    "score": 2.0,
                    "reason": "错误继承了无关历史，或保留了错误的动作/位置/工具参数。",
                },
            }
        )
    return preferences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build memory tool-vs-wrong-tool contrast preference rows.")
    p.add_argument("--tasks", default="data/rl/memory_tasks.jsonl")
    p.add_argument("--output", default="data/rl/memory_contrast_preferences.jsonl")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    rows = build_memory_contrast_preferences(tasks)
    write_jsonl(rows, args.output)
    print(f"[out] preferences={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
