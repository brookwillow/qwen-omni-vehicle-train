#!/usr/bin/env python3
"""Build boundary DPO preference files from an eval report.

This extracts valid-tool requests that were over-predicted as
``NoiseDoNotAct`` or ``Reject`` and writes chosen=GT tool, rejected=bad
boundary decision rows for ``train_memory_dpo_lora.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compact_tool_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": tool_name, "arguments": args or {}}


def is_over_noise(error: dict[str, Any]) -> bool:
    return (
        error.get("expected_type") == "Action"
        and error.get("pred_tool") == "NoiseDoNotAct"
        and isinstance(error.get("gt_tool"), str)
    )


def is_false_reject(error: dict[str, Any]) -> bool:
    return (
        error.get("expected_type") == "Action"
        and error.get("pred_type") == "Reject"
        and isinstance(error.get("gt_tool"), str)
    )


def build_preference_row(
    error: dict[str, Any],
    *,
    task_type: str,
    rejected: Any,
    source_report: str,
    id_prefix: str,
) -> dict[str, Any]:
    file_name = error.get("file", "")
    sample_id = error.get("id", "")
    gt_tool = error["gt_tool"]
    gt_args = error.get("gt_args") or {}
    return {
        "id": f"{id_prefix}_{sample_id}",
        "task_type": task_type,
        "history": error.get("history", []) or [],
        "current_query": error.get("query", ""),
        "chosen": compact_tool_call(gt_tool, gt_args),
        "rejected": rejected,
        "source_eval_key": f"{file_name}#{sample_id}",
        "source_report": source_report,
        "source_err_type": error.get("err_type", ""),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_boundary_preferences(report_path: Path, output_dir: Path) -> tuple[Path, Path, int, int]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors = report.get("errors", [])
    source_report = report_path.name

    over_noise_rows = [
        build_preference_row(
            error,
            task_type="still_over_noise_20260609",
            rejected=compact_tool_call("NoiseDoNotAct", {}),
            source_report=source_report,
            id_prefix="still_over_noise_20260609",
        )
        for error in errors
        if is_over_noise(error)
    ]
    false_reject_rows = [
        build_preference_row(
            error,
            task_type="false_reject_20260609",
            rejected="Reject。",
            source_report=source_report,
            id_prefix="false_reject_20260609",
        )
        for error in errors
        if is_false_reject(error)
    ]

    over_noise_path = output_dir / "still_over_noise_preferences_20260609.jsonl"
    false_reject_path = output_dir / "false_reject_preferences_20260609.jsonl"
    write_jsonl(over_noise_path, over_noise_rows)
    write_jsonl(false_reject_path, false_reject_rows)
    return over_noise_path, false_reject_path, len(over_noise_rows), len(false_reject_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/rl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    over_noise_path, false_reject_path, over_noise_count, false_reject_count = build_boundary_preferences(
        args.report, args.output_dir
    )
    print(f"[write] {over_noise_path} rows={over_noise_count}")
    print(f"[write] {false_reject_path} rows={false_reject_count}")


if __name__ == "__main__":
    main()
