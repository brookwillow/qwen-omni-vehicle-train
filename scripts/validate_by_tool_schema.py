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


def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        args = data.get("arguments")
        return data["name"], args if isinstance(args, dict) else None

    action_match = re.match(r"Action:\s*(\S+)", text)
    if not action_match:
        return None, None
    tool_name = action_match.group(1)
    m = ACTION_INPUT_RE.search(text)
    if not m:
        return tool_name, None
    try:
        args = json.loads(m.group(1))
    except json.JSONDecodeError:
        return tool_name, None
    return tool_name, args if isinstance(args, dict) else None


def is_tool_result_msg(msg: dict) -> bool:
    if msg.get("role") != "tool":
        return False
    content = msg.get("content", "").strip()
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return False
    return True


def is_tts_response_msg(msg: dict) -> bool:
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content", "").strip()
    if not content or content.startswith("Reject"):
        return False
    return not is_action_msg(msg)


def is_action_msg(msg: dict) -> bool:
    if msg.get("role") != "assistant":
        return False
    tool_name, _ = parse_tool_call(msg.get("content", ""))
    return tool_name is not None


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

        # Explicit numeric levels are valid for value-like fields even when the
        # schema also lists common enum levels.
        if key == "value" and (_is_numeric_string(val) or _is_percentage_string(val)):
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
                is_action_msg(msg)
                and parse_tool_call(msg["content"])[0] == "NoiseDoNotAct"
                and msg_idx != len(msgs) - 1
            ):
                file_errors.append(f"  Line {i}: NoiseDoNotAct must be the final message in the sample")
            if (
                is_action_msg(msg)
                and msg_idx + 2 < len(msgs)
                and is_tool_result_msg(msgs[msg_idx + 1])
                and is_tts_response_msg(msgs[msg_idx + 2])
                and msg_idx != 0
            ):
                file_errors.append(
                    f"  Line {i}: tool-role result/TTS response sequence must be a standalone result sample"
                )

        # Validate all assistant messages whose action matches the file's tool name.
        # Multi-turn samples may start with a different tool, then call this tool later.
        found_tool = False
        for msg in msgs:
            if msg["role"] != "assistant":
                continue
            content = msg["content"]
            action_name, params = parse_tool_call(content)
            if not action_name:
                continue
            if action_name != tool_name:
                continue

            found_tool = True

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
