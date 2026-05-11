#!/usr/bin/env python3
"""Validate by_tool JSONL files against the tool schema."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS_JSON = ROOT / "data" / "tools.json"
BY_TOOL_DIR = ROOT / "data" / "splits" / "by_tool"

tools = {t["name"]: t for t in json.loads(TOOLS_JSON.read_text())}

ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)


def extract_action_input(text: str) -> dict | None:
    m = ACTION_INPUT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _is_numeric_string(val) -> bool:
    """Return True if val is a string representing an integer or decimal number."""
    if not isinstance(val, str):
        return False
    try:
        float(val)
        return True
    except ValueError:
        return False


def _is_percentage_string(val) -> bool:
    if not isinstance(val, str):
        return False
    return val.endswith("%") and _is_numeric_string(val[:-1])


def validate_against_schema(tool_name: str, params: dict) -> list[str]:
    errors = []
    schema = tools.get(tool_name, {}).get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required fields
    for req in required:
        if req not in params:
            errors.append(f"Missing required field: {req!r}")

    # Check enum values
    for key, val in params.items():
        if key not in properties:
            errors.append(f"Unknown field: {key!r}")
            continue
        prop = properties[key]
        if "enum" not in prop:
            continue
        if val in prop["enum"]:
            continue

        # ClimateControl: numeric strings are explicitly allowed for temperature/wind
        # per description: "value=[NUMBER STRING OR LEVEL STRING]"
        if tool_name == "ClimateControl" and key == "value" and _is_numeric_string(val):
            continue

        # WindowControl: percentage strings are valid positional values for 开到/关到 actions
        if tool_name == "WindowControl" and key == "value" and _is_percentage_string(val):
            action = params.get("action", "")
            if action in ("开到", "关到", "再开", "再关"):
                continue

        errors.append(f"Invalid enum value for {key!r}: {val!r} not in {prop['enum']}")

    return errors


total_samples = 0
total_errors = 0
files_with_errors = 0

for jsonl_file in sorted(BY_TOOL_DIR.glob("*.jsonl")):
    tool_name = jsonl_file.stem
    if tool_name not in tools:
        print(f"[WARN] No schema found for tool: {tool_name}")
        continue

    lines = jsonl_file.read_text().strip().splitlines()
    file_errors = []

    for i, line in enumerate(lines, 1):
        sample = json.loads(line)
        msgs = sample["messages"]
        for msg_idx, msg in enumerate(msgs):
            if (
                msg["role"] == "assistant"
                and msg["content"].startswith("Action: NoiseDoNotAct")
                and msg_idx != len(msgs) - 1
            ):
                file_errors.append(f"  Line {i}: NoiseDoNotAct must be the final message in the sample")

        # Validate all assistant messages whose action matches the file's tool name.
        # Multi-turn samples may start with a different tool, then call this tool later.
        found_tool = False
        for msg in msgs:
            if msg["role"] != "assistant" or "Action:" not in msg["content"]:
                continue
            content = msg["content"]
            action_match = re.match(r"Action:\s*(\S+)", content)
            if not action_match:
                continue
            action_name = action_match.group(1)
            if action_name != tool_name:
                continue

            found_tool = True

            params = extract_action_input(content)
            if params is None:
                if tool_name == "NoiseDoNotAct":
                    continue
                file_errors.append(f"  Line {i}: Failed to parse Action Input")
                continue

            errs = validate_against_schema(tool_name, params)
            for err in errs:
                file_errors.append(f"  Line {i}: {err}")

        if not found_tool:
            file_errors.append(f"  Line {i}: No {tool_name!r} action found in sample")

    count = len(lines)
    total_samples += count
    if file_errors:
        files_with_errors += 1
        total_errors += len(file_errors)
        print(f"[ERRORS] {tool_name}.jsonl ({count} samples):")
        for e in file_errors:
            print(e)
    else:
        status = "OK" if count >= 15 else f"LOW ({count} samples)"
        print(f"[{status}] {tool_name}.jsonl ({count} samples)")

print(f"\nTotal: {total_samples} samples, {total_errors} errors across {files_with_errors} files")
