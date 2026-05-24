#!/usr/bin/env python3
"""Report tool schema coverage across SFT splits, eval sets, and RL data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_rl_schema import DEFAULT_RL_DATASETS, iter_tool_calls
from scripts.validate_splits import load_schema, parse_actions


def canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tool_call_key(tool_name: str, args: dict[str, Any]) -> str:
    return f"{tool_name}:{canonical_args(args)}"


def iter_sft_calls(splits_dir: Path) -> list[tuple[str, str, dict[str, Any], str]]:
    calls = []
    for file in sorted(splits_dir.rglob("*.jsonl")):
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for msg in row.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                for tool_name, args in parse_actions(str(msg.get("content", ""))):
                    if isinstance(args, dict):
                        calls.append(("sft", tool_name, args, f"{file}:{line_no}"))
    return calls


def iter_eval_calls(eval_dir: Path) -> list[tuple[str, str, dict[str, Any], str]]:
    calls = []
    for file in sorted(eval_dir.glob("*_test.json")):
        rows = json.loads(file.read_text(encoding="utf-8"))
        for index, row in enumerate(rows, start=1):
            for call in row.get("expected_tool_calls", []) or []:
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    args = call.get("arguments", {})
                    if isinstance(args, dict):
                        calls.append(("eval", call["name"], args, f"{file}:{row.get('id', index)}"))
    return calls


def iter_rl_calls(rl_dir: Path) -> list[tuple[str, str, dict[str, Any], str]]:
    calls = []
    for name in DEFAULT_RL_DATASETS:
        file = rl_dir / name
        if not file.exists():
            continue
        for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for tool_name, args, _ in iter_tool_calls(row, "$"):
                if isinstance(args, dict):
                    calls.append(("rl", tool_name, args, f"{file}:{line_no}"))
    return calls


def build_coverage(
    calls: list[tuple[str, str, dict[str, Any], str]],
    schema: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_source_tool: dict[str, Counter[str]] = defaultdict(Counter)
    param_values: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    combos: dict[str, Counter[str]] = defaultdict(Counter)
    combo_examples: dict[str, dict[str, str]] = defaultdict(dict)

    for source, tool_name, args, origin in calls:
        by_source_tool[source][tool_name] += 1
        key = tool_call_key(tool_name, args)
        combos[source][key] += 1
        combo_examples[source].setdefault(key, origin)
        for param, value in args.items():
            param_values[source][tool_name][param][str(value)] += 1

    missing_enum_values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in ("sft", "eval", "rl"):
        for tool_name, tool_schema in schema.items():
            props = tool_schema["props"]
            for param, prop in props.items():
                enum = prop.get("enum")
                if not enum:
                    continue
                seen = set(param_values[source][tool_name][param])
                missing = [value for value in enum if value not in seen]
                if missing:
                    missing_enum_values[source].append(
                        {
                            "tool": tool_name,
                            "param": param,
                            "seen": len(seen),
                            "total": len(enum),
                            "missing": missing,
                        }
                    )

    eval_missing_in_sft = []
    for key, count in combos["eval"].most_common():
        if combos["sft"][key] == 0:
            tool_name, raw_args = key.split(":", 1)
            eval_missing_in_sft.append(
                {
                    "tool": tool_name,
                    "args": json.loads(raw_args),
                    "eval_count": count,
                    "example": combo_examples["eval"].get(key, ""),
                }
            )

    rl_missing_in_sft = []
    for key, count in combos["rl"].most_common():
        if combos["sft"][key] == 0:
            tool_name, raw_args = key.split(":", 1)
            rl_missing_in_sft.append(
                {
                    "tool": tool_name,
                    "args": json.loads(raw_args),
                    "rl_count": count,
                    "example": combo_examples["rl"].get(key, ""),
                }
            )

    return {
        "source_tool_counts": {source: dict(counter.most_common()) for source, counter in by_source_tool.items()},
        "missing_enum_values": dict(missing_enum_values),
        "eval_combos_missing_in_sft": eval_missing_in_sft,
        "rl_combos_missing_in_sft": rl_missing_in_sft,
        "param_values": {
            source: {
                tool: {param: dict(values.most_common()) for param, values in params.items()}
                for tool, params in tools.items()
            }
            for source, tools in param_values.items()
        },
    }


def infer_target_split(tool_name: str) -> str:
    return f"data/splits/hard_cases/{tool_name}_coverage.jsonl"


def _combo_priority(item: dict[str, Any]) -> str:
    count = int(item.get("eval_count") or item.get("rl_count") or 0)
    args = item.get("args") or {}
    if count >= 3:
        return "P0"
    if any(param in args for param in ("action", "device", "feature", "value")):
        return "P1"
    return "P2"


def build_hard_case_backlog(report: dict[str, Any], limit: int = 80) -> list[dict[str, Any]]:
    """Convert coverage gaps into grouped hard-case tasks."""
    grouped: dict[str, dict[str, Any]] = {}

    for item in report.get("eval_combos_missing_in_sft", []):
        tool = item["tool"]
        bucket = grouped.setdefault(
            f"eval_missing_combo:{tool}",
            {
                "priority": "P2",
                "issue": f"eval_missing_combo:{tool}",
                "source": "eval",
                "tool": tool,
                "target_split": infer_target_split(tool),
                "recommendation": "补充非评估原句的同 schema 组合 hard case，并用 --sample-weight 对 hard_cases 加权。",
                "items": [],
            },
        )
        bucket["items"].append(item)
        if _combo_priority(item) == "P0":
            bucket["priority"] = "P0"
        elif bucket["priority"] != "P0" and _combo_priority(item) == "P1":
            bucket["priority"] = "P1"

    for item in report.get("rl_combos_missing_in_sft", []):
        tool = item["tool"]
        bucket = grouped.setdefault(
            f"rl_missing_combo:{tool}",
            {
                "priority": "P2",
                "issue": f"rl_missing_combo:{tool}",
                "source": "rl",
                "tool": tool,
                "target_split": infer_target_split(tool),
                "recommendation": "RL chosen/rejected 中出现但 SFT 未覆盖，优先补 SFT 基础样本，再继续做 DPO。",
                "items": [],
            },
        )
        bucket["items"].append(item)
        if bucket["priority"] != "P0":
            bucket["priority"] = "P1"

    for source, rows in report.get("missing_enum_values", {}).items():
        if source != "sft":
            continue
        for row in rows:
            if not row.get("missing"):
                continue
            tool = row["tool"]
            param = row["param"]
            key = f"sft_missing_enum:{tool}.{param}"
            grouped[key] = {
                "priority": "P1" if row["seen"] == 0 else "P2",
                "issue": key,
                "source": "schema",
                "tool": tool,
                "target_split": infer_target_split(tool),
                "recommendation": "补充该参数枚举值的自然表达，但只补业务上真实可触发且不歧义的值。",
                "items": [row],
            }

    priority_rank = {"P0": 0, "P1": 1, "P2": 2}
    return sorted(
        grouped.values(),
        key=lambda item: (priority_rank.get(item["priority"], 9), item["tool"], item["issue"]),
    )[:limit]


def write_backlog_markdown(backlog: list[dict[str, Any]], path: Path, per_issue_limit: int = 10) -> None:
    lines = ["# Schema Coverage Hard Case Backlog", ""]
    lines.append("补数据原则：不要直接复制 eval 原句；围绕相同 schema 组合生成 2-5 条非评估原句强对比样本。")
    lines.append("")
    for item in backlog:
        lines.append(f"## {item['priority']} {item['issue']}")
        lines.append(f"- Tool: `{item['tool']}`")
        lines.append(f"- Target split: `{item['target_split']}`")
        lines.append(f"- Recommendation: {item['recommendation']}")
        lines.append("- Coverage gaps:")
        for gap in item["items"][:per_issue_limit]:
            if "args" in gap:
                count = gap.get("eval_count", gap.get("rl_count", 0))
                lines.append(f"  - args={gap['args']} count={count} example={gap.get('example', '')}")
            else:
                missing = ", ".join(map(str, gap.get("missing", [])[:16]))
                suffix = "" if len(gap.get("missing", [])) <= 16 else f" ... (+{len(gap['missing']) - 16})"
                lines.append(f"  - `{gap['tool']}.{gap['param']}` seen={gap['seen']}/{gap['total']} missing={missing}{suffix}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path, limit: int) -> None:
    lines = ["# Schema Coverage Report", ""]
    lines.append("## Source Tool Counts")
    for source, counts in report["source_tool_counts"].items():
        lines.append(f"### {source}")
        for tool, count in list(counts.items())[:limit]:
            lines.append(f"- `{tool}`: {count}")
        lines.append("")

    lines.append("## Eval Combos Missing In SFT")
    for item in report["eval_combos_missing_in_sft"][:limit]:
        lines.append(f"- `{item['tool']}` {item['args']} eval_count={item['eval_count']} example={item['example']}")
    lines.append("")

    lines.append("## RL Combos Missing In SFT")
    for item in report["rl_combos_missing_in_sft"][:limit]:
        lines.append(f"- `{item['tool']}` {item['args']} rl_count={item['rl_count']} example={item['example']}")
    lines.append("")

    lines.append("## Missing Enum Values")
    for source, rows in report["missing_enum_values"].items():
        lines.append(f"### {source}")
        for row in sorted(rows, key=lambda r: (r["tool"], r["param"]))[:limit]:
            missing = ", ".join(map(str, row["missing"][:12]))
            suffix = "" if len(row["missing"]) <= 12 else f" ... (+{len(row['missing']) - 12})"
            lines.append(f"- `{row['tool']}.{row['param']}` seen={row['seen']}/{row['total']} missing={missing}{suffix}")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", default=str(ROOT / "data" / "tools.json"))
    parser.add_argument("--splits-dir", default=str(ROOT / "data" / "splits"))
    parser.add_argument("--eval-dir", default=str(ROOT / "data" / "eval"))
    parser.add_argument("--rl-dir", default=str(ROOT / "data" / "rl"))
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument("--output-md", help="Write a Markdown report")
    parser.add_argument("--backlog-md", help="Write grouped hard-case backlog markdown")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    schema = load_schema(args.tools)
    calls = []
    calls.extend(iter_sft_calls(Path(args.splits_dir)))
    calls.extend(iter_eval_calls(Path(args.eval_dir)))
    calls.extend(iter_rl_calls(Path(args.rl_dir)))
    report = build_coverage(calls, schema)

    if args.output_md:
        write_markdown(report, Path(args.output_md), args.limit)
        print(f"[coverage] wrote markdown -> {args.output_md}")
    if args.backlog_md:
        backlog = build_hard_case_backlog(report, args.limit)
        write_backlog_markdown(backlog, Path(args.backlog_md))
        print(f"[coverage] wrote backlog -> {args.backlog_md} ({len(backlog)} items)")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print("# Schema Coverage Summary")
    for source, counts in report["source_tool_counts"].items():
        total = sum(counts.values())
        print(f"{source}: total_calls={total} tools={len(counts)}")
    print(f"eval_combos_missing_in_sft={len(report['eval_combos_missing_in_sft'])}")
    print(f"rl_combos_missing_in_sft={len(report['rl_combos_missing_in_sft'])}")
    for source, rows in report["missing_enum_values"].items():
        print(f"{source}_missing_enum_slots={len(rows)}")


if __name__ == "__main__":
    main()
