#!/usr/bin/env python3
"""Validate RL JSONL artifacts against the current tool schema.

The RL files contain tool calls in several shapes: direct chosen/rejected JSON,
expected.target_tool_call, relevant_memory.tool_call, and sometimes JSON strings.
This script recursively extracts those tool-call-like values and validates
tool name, required fields, unknown fields, and enum values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
NUMERIC_VALUE_RE = re.compile(r"^\d+(\.\d+)?%?$")
DEFAULT_RL_DATASETS = (
    "memory_tasks.jsonl",
    "memory_preferences.jsonl",
    "memory_contrast_preferences.jsonl",
    "tool_tts_preferences.jsonl",
    "tool_boundary_preferences.jsonl",
    "current_noise_boundary_preferences.jsonl",
    "noise_false_positive_preferences.jsonl",
    "anti_over_noise_preferences.jsonl",
    "still_over_noise_preferences_round2.jsonl",
    "still_over_noise_preferences_round3.jsonl",
    "still_over_noise_preferences_20260609.jsonl",
    "still_over_noise_preferences_20260609_round2.jsonl",
    "false_reject_preferences_20260609.jsonl",
    "wrong_tool_preferences.jsonl",
    "false_reject_clarify_preferences.jsonl",
    "extra_args_preferences.jsonl",
)


def load_schema(tools_path: Path) -> dict[str, dict[str, Any]]:
    tools = json.loads(tools_path.read_text(encoding="utf-8"))
    schema = {}
    for tool in tools:
        fn = tool.get("function", tool)
        params = fn.get("parameters") or fn.get("inputSchema") or {}
        schema[fn["name"]] = {
            "props": params.get("properties", {}),
            "required": params.get("required", []),
        }
    return schema


def parse_jsonish_string(value: str) -> Any:
    text = value.strip()
    if not text or text in {"Reject", "NoiseDoNotAct"}:
        return None
    if not text.startswith(("{", "[")):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def iter_tool_calls(value: Any, path: str) -> list[tuple[str, dict[str, Any] | None, str]]:
    calls: list[tuple[str, dict[str, Any] | None, str]] = []

    if isinstance(value, str):
        parsed = parse_jsonish_string(value)
        if parsed is not None:
            calls.extend(iter_tool_calls(parsed, path))
        return calls

    if isinstance(value, list):
        for idx, item in enumerate(value):
            calls.extend(iter_tool_calls(item, f"{path}[{idx}]"))
        return calls

    if not isinstance(value, dict):
        return calls

    if isinstance(value.get("name"), str) and "arguments" in value:
        args = value.get("arguments")
        calls.append((value["name"], args if isinstance(args, dict) else None, path))
        return calls

    for key, child in value.items():
        calls.extend(iter_tool_calls(child, f"{path}.{key}"))
    return calls


def allows_schema_free_value(tool_name: str, param: str, val: Any, args: dict[str, Any], prop: dict[str, Any]) -> bool:
    if not isinstance(val, str):
        return False

    if tool_name == "ClimateControl" and param == "value" and NUMERIC_VALUE_RE.match(val):
        return True

    if tool_name == "WindowControl" and param == "value" and val.endswith("%"):
        action = args.get("action", "")
        return action in {"开到", "关到", "再开", "再关"} and NUMERIC_VALUE_RE.match(val[:-1]) is not None

    desc = prop.get("description", "")
    return param == "value" and NUMERIC_VALUE_RE.match(val) is not None and (
        "numeric" in desc or "percentage" in desc
    )


def validate_tool_call(
    tool_name: str,
    args: dict[str, Any] | None,
    path: str,
    schema: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if tool_name not in schema:
        return [{"type": "UNKNOWN_TOOL", "path": path, "tool": tool_name}]

    if tool_name == "NoiseDoNotAct":
        return issues

    if args is None:
        return [{"type": "INVALID_ARGUMENTS", "path": path, "tool": tool_name}]

    tool_schema = schema[tool_name]
    props = tool_schema["props"]
    required = tool_schema["required"]

    for req in required:
        if req not in args:
            issues.append({"type": "MISSING_REQUIRED", "path": path, "tool": tool_name, "param": req, "args": args})

    for param, val in args.items():
        if param not in props:
            issues.append({"type": "UNKNOWN_PARAM", "path": path, "tool": tool_name, "param": param, "args": args})
            continue
        prop = props[param]
        enum = prop.get("enum")
        if enum is None or val in enum:
            continue
        if allows_schema_free_value(tool_name, param, val, args, prop):
            continue
        issues.append(
            {
                "type": "INVALID_ENUM",
                "path": path,
                "tool": tool_name,
                "param": param,
                "value": val,
                "allowed": enum,
                "args": args,
            }
        )

    return issues


def validate_file(path: Path, schema: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for tool_name, args, call_path in iter_tool_calls(row, "$"):
            for issue in validate_tool_call(tool_name, args, call_path, schema):
                issue["file"] = str(path)
                issue["line"] = line_no
                issues.append(issue)
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-dir", default=str(ROOT / "data" / "rl"))
    parser.add_argument("--tools", default=str(ROOT / "data" / "tools.json"))
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional explicit JSONL files. Defaults to stable RL training datasets.",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Also validate generated candidate/audit JSONL artifacts under --rl-dir.",
    )
    args = parser.parse_args()

    schema = load_schema(Path(args.tools))
    if args.files:
        files = [Path(p) for p in args.files]
    elif args.include_artifacts:
        files = sorted(Path(args.rl_dir).glob("*.jsonl"))
    else:
        files = [Path(args.rl_dir) / name for name in DEFAULT_RL_DATASETS]

    all_issues: list[dict[str, Any]] = []
    for file in files:
        file_issues = validate_file(file, schema)
        all_issues.extend(file_issues)

    print("=" * 60)
    print(f"RL schema issues: {len(all_issues)}")
    print("=" * 60)

    for issue_type, count in Counter(i["type"] for i in all_issues).most_common():
        print(f"  {issue_type:<20} {count}")

    for file, count in Counter(i["file"] for i in all_issues).most_common():
        print(f"  {Path(file).name:<35} {count}")

    for issue in all_issues[:40]:
        shown = {k: v for k, v in issue.items() if k != "allowed"}
        print(json.dumps(shown, ensure_ascii=False))

    if all_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
