#!/usr/bin/env python3
"""Build direct preferences for tool JSON output over TTS completion text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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


def position_prefix(args: dict[str, Any]) -> str:
    position = args.get("position")
    return f"{position}" if isinstance(position, str) and position else ""


def render_tts_from_tool_call(tool: dict[str, Any]) -> str:
    name = tool.get("name")
    args = tool.get("arguments") if isinstance(tool.get("arguments"), dict) else {}
    action = str(args.get("action", "设置"))
    device = str(args.get("device", ""))
    position = position_prefix(args)

    if name == "WindowControl":
        return f"好的，已为您{action}{position}车窗。"
    if name == "LightControl":
        light = device or "灯光"
        return f"好的，已为您{action}{light}。"
    if name == "GateControl":
        gate = f"{position}{device}" if device else f"{position}车门"
        return f"好的，已为您{action}{gate}。"
    if name == "SeatControl":
        feature = args.get("feature")
        target = f"{position}座椅"
        if feature:
            target += str(feature)
        return f"好的，已为您{action}{target}。"
    if name == "ClimateControl":
        value = args.get("value")
        feature = args.get("feature")
        if value:
            return f"好的，已将空调{action}{value}。"
        if feature:
            return f"好的，已为您{action}空调{feature}。"
        return f"好的，已为您{action}空调。"
    if name == "VoiceControl":
        feature = str(args.get("feature", "音量"))
        return f"好的，已为您{action}{feature}。"
    return "好的，已经为您处理。"


def target_tool_call(task: dict[str, Any]) -> dict[str, Any] | None:
    expected = task.get("expected") if isinstance(task.get("expected"), dict) else {}
    tool = expected.get("target_tool_call")
    if not isinstance(tool, dict):
        return None
    if tool.get("name") == "NoiseDoNotAct":
        return None
    if not isinstance(tool.get("arguments"), dict):
        return None
    return tool


def build_tool_tts_preferences(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferences = []
    seen = set()
    for task in tasks:
        tool = target_tool_call(task)
        if not tool:
            continue
        rejected = render_tts_from_tool_call(tool)
        key = json.dumps(
            {
                "history": task.get("history", []),
                "current_query": task.get("current_query", ""),
                "chosen": tool,
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
                "id": f"{task.get('id', '')}_tool_vs_tts",
                "task_type": "tool_vs_tts",
                "source_task_type": task.get("task_type", ""),
                "history": task.get("history", []),
                "current_query": task.get("current_query", ""),
                "chosen": tool,
                "rejected": rejected,
                "chosen_score": 10.0,
                "rejected_score": 4.0,
                "chosen_source": "expected_tool_json",
                "rejected_source": "synthetic_tts_reply",
                "chosen_verdict": {
                    "tool_or_response_correct": True,
                    "arguments_correct": True,
                    "score": 10.0,
                    "reason": "当前轮需要工具调用，正确输出紧凑工具 JSON。",
                },
                "rejected_verdict": {
                    "tool_or_response_correct": False,
                    "arguments_correct": False,
                    "score": 4.0,
                    "reason": "当前轮需要工具调用，但候选只输出执行完成话术，没有输出工具 JSON。",
                },
            }
        )
    return preferences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tool-JSON-vs-TTS output contract preference rows.")
    p.add_argument("--tasks", default="data/rl/memory_tasks.jsonl")
    p.add_argument("--output", default="data/rl/tool_tts_preferences.jsonl")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    rows = build_tool_tts_preferences(tasks)
    write_jsonl(rows, args.output)
    print(f"[out] preferences={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
