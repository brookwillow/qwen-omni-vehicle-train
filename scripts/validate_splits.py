#!/usr/bin/env python3
"""
Validate all training split JSONL files against the tool schema.
Checks: unknown tools, missing required params, invalid enum values, unknown params.

Usage:
    python scripts/validate_splits.py
    python scripts/validate_splits.py --fix   # auto-remove unfixable samples (with backup)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


NUMERIC_VALUE_RE = re.compile(r"^\d+(\.\d+)?%?$")


def load_schema(tools_path: str = "data/tools.json") -> dict:
    with open(tools_path) as f:
        tools = json.load(f)
    schema = {}
    for t in tools:
        fn = t.get("function", t)
        name = fn["name"]
        params = fn.get("parameters") or fn.get("inputSchema") or {}
        props = params.get("properties", {})
        required = params.get("required", [])
        schema[name] = {"props": props, "required": required}
    return schema


def parse_actions(content: str):
    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        args = data.get("arguments")
        return [(data["name"], args if isinstance(args, dict) else None)]
    if isinstance(data, list):
        actions = []
        for item in data:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or "arguments" not in item
            ):
                return []
            args = item.get("arguments")
            actions.append((item["name"], args if isinstance(args, dict) else None))
        return actions

    m = re.match(r"Action:\s*(\S+)\s*\nAction Input:\s*(\{.*\})", content, re.DOTALL)
    if not m:
        return []
    tool_name = m.group(1).strip()
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return [(tool_name, None)]
    return [(tool_name, args)]


def allows_free_numeric(param: str, prop_def: dict) -> bool:
    desc = prop_def.get("description", "")
    return "numeric" in desc or "percentage" in desc or param == "value"


def validate_sample(sample: dict, source: str, schema: dict) -> list[dict]:
    issues = []
    tool_names = set(schema.keys())
    msgs = sample.get("messages", [])

    for msg in msgs:
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        if not (content.startswith("Action:") or content.lstrip().startswith(("{", "["))):
            continue

        actions = parse_actions(content)

        if not actions:
            if content.lstrip().startswith("["):
                try:
                    data = json.loads(content.strip())
                except json.JSONDecodeError:
                    data = None
                is_toolish_array = isinstance(data, list) and any(
                    isinstance(item, dict)
                    and (item.get("name") in tool_names or "arguments" in item)
                    for item in data
                )
                if not is_toolish_array:
                    continue
            issues.append({"type": "PARSE_FAIL", "source": source, "snippet": content[:100]})
            continue
        for tool_name, args in actions:
            if tool_name not in tool_names:
                issues.append({"type": "UNKNOWN_TOOL", "source": source, "tool": tool_name, "snippet": content[:100]})
                continue
            if args is None:
                issues.append({"type": "JSON_PARSE_FAIL", "source": source, "tool": tool_name, "snippet": content[:100]})
                continue

            s = schema[tool_name]
            props = s["props"]
            required = s["required"]

            for req in required:
                if req not in args:
                    issues.append({"type": "MISSING_REQUIRED", "source": source, "tool": tool_name, "param": req, "args": args})

            for param, val in args.items():
                if param not in props:
                    issues.append({"type": "UNKNOWN_PARAM", "source": source, "tool": tool_name, "param": param, "args": args})
                    continue
                prop_def = props[param]
                if "enum" in prop_def and isinstance(val, str):
                    allows_free = allows_free_numeric(param, prop_def)
                    if val not in prop_def["enum"]:
                        if not allows_free:
                            issues.append({"type": "INVALID_ENUM", "source": source, "tool": tool_name,
                                           "param": param, "val": val,
                                           "allowed": prop_def["enum"], "args": args})
                        elif not NUMERIC_VALUE_RE.match(val):
                            issues.append({"type": "INVALID_ENUM_FREE", "source": source, "tool": tool_name,
                                           "param": param, "val": val, "args": args})
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--tools", default="data/tools.json")
    parser.add_argument("--fix", action="store_true", help="Remove invalid samples (backs up originals)")
    args = parser.parse_args()

    schema = load_schema(args.tools)
    all_errors: list[dict] = []
    # {filepath: [bad line indices]}
    bad_lines: dict[str, list[int]] = defaultdict(list)

    for split_file in sorted(Path(args.splits_dir).rglob("*.jsonl")):
        with open(split_file) as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            issues = validate_sample(sample, split_file.name, schema)
            if issues:
                for iss in issues:
                    iss["line"] = idx + 1
                all_errors.extend(issues)
                bad_lines[str(split_file)].append(idx)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Total issues found: {len(all_errors)}")
    print(f"{'='*60}")

    type_counts = Counter(e["type"] for e in all_errors)
    for etype, cnt in type_counts.most_common():
        print(f"  {etype:<25} {cnt}")

    # Per file
    print(f"\nPer file:")
    file_counts = Counter(e["source"] for e in all_errors)
    for fname, cnt in file_counts.most_common():
        bad = len(set(e["line"] for e in all_errors if e["source"] == fname))
        print(f"  {fname:<30} {cnt} issues in {bad} samples")

    # Details grouped by type
    print(f"\n{'='*60}")
    print("Sample details (up to 8 per type):")
    print(f"{'='*60}")
    seen: dict[str, int] = {}
    for e in all_errors:
        key = e["type"]
        if seen.get(key, 0) >= 8:
            continue
        seen[key] = seen.get(key, 0) + 1
        print(json.dumps({k: v for k, v in e.items() if k != "allowed"}, ensure_ascii=False))

    # Show allowed enums for INVALID_ENUM separately to keep output readable
    invalid_enum_errs = [e for e in all_errors if e["type"] == "INVALID_ENUM"]
    if invalid_enum_errs:
        print(f"\nINVALID_ENUM details (tool / param / bad_value / allowed):")
        for e in invalid_enum_errs[:20]:
            print(f"  {e['tool']}.{e['param']} = {e['val']!r}  allowed={e['allowed']}")

    # ---- Fix ----
    if args.fix and bad_lines:
        print(f"\n{'='*60}")
        print("--fix mode: removing invalid samples")
        total_removed = 0
        for filepath, bad_idxs in bad_lines.items():
            p = Path(filepath)
            backup = p.with_suffix(".jsonl.bak_validate")
            shutil.copy(p, backup)
            with open(p) as f:
                lines = f.readlines()
            kept = [line for i, line in enumerate(lines) if i not in set(bad_idxs)]
            with open(p, "w") as f:
                f.writelines(kept)
            removed = len(lines) - len(kept)
            total_removed += removed
            print(f"  {p.name}: removed {removed} samples (backup: {backup.name})")
        print(f"Total removed: {total_removed}")
    elif not args.fix and bad_lines:
        print(f"\nRun with --fix to auto-remove invalid samples.")


if __name__ == "__main__":
    main()
