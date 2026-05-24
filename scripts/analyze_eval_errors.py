#!/usr/bin/env python3
"""Cluster eval report errors into actionable buckets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _required_fields_by_tool(tools_path: str | Path) -> dict[str, set[str]]:
    tools = json.loads(Path(tools_path).read_text(encoding="utf-8"))
    return {
        tool["name"]: set(tool.get("inputSchema", {}).get("required", []))
        for tool in tools
        if "name" in tool
    }


def arg_delta(gt_args: dict[str, Any], pred_args: dict[str, Any]) -> dict[str, Any]:
    """Return missing/extra/changed arg slots for one tool-call comparison."""
    gt_keys = set(gt_args)
    pred_keys = set(pred_args)
    changed = {
        key: {"gt": gt_args.get(key), "pred": pred_args.get(key)}
        for key in sorted(gt_keys & pred_keys)
        if gt_args.get(key) != pred_args.get(key)
    }
    return {
        "missing": sorted(gt_keys - pred_keys),
        "extra": sorted(pred_keys - gt_keys),
        "changed": changed,
    }


def classify_error(error: dict[str, Any], required_by_tool: dict[str, set[str]] | None = None) -> str:
    """Return a stable issue label for one eval error."""
    required_by_tool = required_by_tool or {}
    err_type = error.get("err_type")
    expected_type = error.get("expected_type")
    pred_type = error.get("pred_type")
    gt_tool = error.get("gt_tool")
    pred_tool = error.get("pred_tool")
    gt_args = error.get("gt_args") or {}
    pred_args = error.get("pred_args") or {}
    query = error.get("query") or ""
    pred_raw = error.get("pred_raw") or ""

    if err_type == "type-err":
        if expected_type == "Action" and pred_type == "Reject":
            return "over_reject"
        if expected_type == "Action" and pred_type == "Clarify":
            if pred_raw and not any(token in pred_raw for token in ("请问", "哪", "什么", "?","？")):
                return "tool_vs_tts_contract"
            return "over_clarify"
        if expected_type == "Action" and pred_type == "ParseFail":
            return "parse_fail_action"
        if expected_type in {"Reject", "Clarify"} and pred_type == "Action":
            return "over_action"
        return "type_mismatch"

    if err_type == "tool-err":
        if pred_tool == "NoiseDoNotAct":
            return "over_noise"
        if pred_tool in {None, ""}:
            return "missing_tool"
        return f"tool_confusion:{pred_tool}->{gt_tool}"

    if err_type == "args-err" and isinstance(gt_args, dict) and isinstance(pred_args, dict):
        delta = arg_delta(gt_args, pred_args)
        missing = set(delta["missing"])
        extra = set(delta["extra"])
        changed = set(delta["changed"])
        required_missing = missing & required_by_tool.get(gt_tool, set())

        if required_missing:
            return "missing_required_args:" + ",".join(sorted(required_missing))
        if missing and not extra and not changed:
            return "missing_optional_args:" + ",".join(sorted(missing))
        if extra and not missing and not changed:
            return "extra_args:" + ",".join(sorted(extra))
        if changed and not missing and not extra:
            return "wrong_arg_value:" + ",".join(sorted(changed))
        if missing or extra or changed:
            parts = []
            if missing:
                parts.append("missing=" + ",".join(sorted(missing)))
            if extra:
                parts.append("extra=" + ",".join(sorted(extra)))
            if changed:
                parts.append("changed=" + ",".join(sorted(changed)))
            return "mixed_arg_error:" + ";".join(parts)

    if "多意图" in query or len(error.get("expected_tool_calls", []) or []) > 1:
        return "multi_intent_scoring"
    return err_type or "unknown"


def summarize_report(report: dict[str, Any], tools_path: str | Path = "data/tools.json") -> dict[str, Any]:
    required_by_tool = _required_fields_by_tool(tools_path)
    errors = report.get("errors", []) or []
    issue_counts: Counter[str] = Counter()
    by_gt_tool: Counter[str] = Counter()
    by_pred_tool: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    confusion_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for error in errors:
        issue = classify_error(error, required_by_tool)
        issue_counts[issue] += 1
        by_gt_tool[str(error.get("gt_tool") or "-")] += 1
        by_pred_tool[str(error.get("pred_tool") or error.get("pred_type") or "-")] += 1
        by_file[str(error.get("file") or "-")] += 1
        by_category[str(error.get("category") or "-")] += 1
        gt_args = error.get("gt_args") or {}
        pred_args = error.get("pred_args") or {}
        if isinstance(gt_args, dict) and isinstance(pred_args, dict):
            delta = arg_delta(gt_args, pred_args)
            for param in delta["missing"]:
                slot_counts[f"missing:{error.get('gt_tool')}:{param}"] += 1
            for param in delta["extra"]:
                slot_counts[f"extra:{error.get('gt_tool')}:{param}"] += 1
            for param, values in delta["changed"].items():
                slot_counts[f"changed:{error.get('gt_tool')}:{param}"] += 1
                confusion_counts[
                    f"{error.get('gt_tool')}.{param}:{values['gt']} -> {values['pred']}"
                ] += 1
        if error.get("err_type") == "tool-err":
            confusion_counts[f"tool:{error.get('gt_tool')} <- {error.get('pred_tool')}"] += 1
        if len(issue_examples[issue]) < 5:
            issue_examples[issue].append(
                {
                    "id": error.get("id"),
                    "file": error.get("file"),
                    "query": error.get("query"),
                    "gt_tool": error.get("gt_tool"),
                    "gt_args": error.get("gt_args"),
                    "pred_type": error.get("pred_type"),
                    "pred_tool": error.get("pred_tool"),
                    "pred_args": error.get("pred_args"),
                    "arg_delta": arg_delta(gt_args, pred_args) if isinstance(gt_args, dict) and isinstance(pred_args, dict) else {},
                }
            )

    return {
        "overall": report.get("overall", {}),
        "error_count": len(errors),
        "issue_counts": dict(issue_counts.most_common()),
        "by_gt_tool": dict(by_gt_tool.most_common()),
        "by_pred_tool": dict(by_pred_tool.most_common()),
        "by_file": dict(by_file.most_common()),
        "by_category": dict(by_category.most_common()),
        "slot_counts": dict(slot_counts.most_common()),
        "confusion_counts": dict(confusion_counts.most_common()),
        "examples": dict(issue_examples),
    }


def _training_focus(issue: str) -> str:
    if issue == "tool_vs_tts_contract":
        return "工具调用与TTS输出契约"
    if issue == "parse_fail_action":
        return "JSON格式与输出稳定性"
    if issue.startswith("wrong_arg_value") or issue.startswith("mixed_arg_error"):
        return "参数值强对比"
    if issue.startswith("missing_") or issue.startswith("extra_args"):
        return "参数槽位边界"
    if issue.startswith("tool_confusion"):
        return "工具边界强对比"
    if issue in {"over_noise", "over_reject", "over_clarify", "over_action"}:
        return "类型边界与拒识保守性"
    return "错误族泛化样本"


def _priority(issue: str, count: int) -> str:
    if count >= 30 and (
        issue.startswith("wrong_arg_value")
        or issue.startswith("mixed_arg_error")
        or issue.startswith("tool_confusion")
        or issue == "over_noise"
        or issue == "tool_vs_tts_contract"
    ):
        return "P0"
    if count >= 10:
        return "P1"
    return "P2"


def build_training_backlog(summary: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    """Convert clustered eval errors into training-data tasks."""
    rows = []
    issue_counts = summary.get("issue_counts", {}) or {}
    examples_by_issue = summary.get("examples", {}) or {}
    for issue, count in list(issue_counts.items())[:limit]:
        examples = examples_by_issue.get(issue, [])[:3]
        gt_tools = Counter(str(example.get("gt_tool") or "-") for example in examples)
        rows.append(
            {
                "priority": _priority(issue, int(count)),
                "issue": issue,
                "count": int(count),
                "training_focus": _training_focus(issue),
                "primary_gt_tool": gt_tools.most_common(1)[0][0] if gt_tools else "-",
                "recommendation": _recommend_training_action(issue),
                "examples": examples,
            }
        )
    return rows


def _recommend_training_action(issue: str) -> str:
    if issue == "tool_vs_tts_contract":
        return "补充同 query 的工具 JSON chosen 与执行完成 TTS rejected 强对比偏好；SFT 中保持 user -> tool JSON 与 tool-result -> TTS 分离。"
    if issue == "parse_fail_action":
        return "补充紧凑 JSON 输出样本和非法格式 rejected 偏好，降低 markdown/解释文本/残缺 JSON。"
    if issue.startswith("wrong_arg_value:action"):
        return "为同一 query 槽位补充打开/关闭/调到/调高/调低/再开/再关/开到/关到的强对比样本。"
    if issue.startswith("wrong_arg_value:value"):
        return "补充 schema 枚举值归一化样本，特别是最高/最低/较高/较低/其他/标准模式等。"
    if issue.startswith("missing_optional_args") or issue.startswith("extra_args"):
        return "补充槽位有无边界样本，让模型学习何时输出 position/value/device/feature，何时省略。"
    if issue.startswith("tool_confusion"):
        return "补充相邻工具强对比样本，使用相似话术但不同工具标签。"
    if issue == "over_noise":
        return "补充短指令和 ASR 口语化动作样本，避免把低信息但明确的车控指令判成 Noise。"
    if issue == "over_reject":
        return "补充工具范围内的自然表达样本，避免工具内请求被拒识。"
    return "按该错误族补充非评估原句的泛化 hard case。"


def format_training_backlog_markdown(backlog: list[dict[str, Any]]) -> str:
    lines = ["# Eval Error Training Backlog", ""]
    for item in backlog:
        lines.append(f"## {item['priority']} {item['issue']} ({item['count']})")
        lines.append(f"- Focus: {item['training_focus']}")
        lines.append(f"- Primary GT tool in examples: {item['primary_gt_tool']}")
        lines.append(f"- Recommendation: {item['recommendation']}")
        if item["examples"]:
            lines.append("- Examples:")
            for example in item["examples"]:
                lines.append(
                    f"  - {example.get('query')} | gt={example.get('gt_tool')} {example.get('gt_args')} "
                    f"| pred={example.get('pred_tool')} {example.get('pred_args')}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _print_counter(title: str, values: dict[str, int], limit: int) -> None:
    print(f"\n## {title}")
    for key, count in list(values.items())[:limit]:
        print(f"{count:4d}  {key}")


def print_summary(summary: dict[str, Any], limit: int) -> None:
    overall = summary.get("overall", {})
    print("# Eval Error Analysis")
    print(f"total={overall.get('total')} errors={summary['error_count']}")
    for metric in ("type_acc", "tool_acc", "args_em"):
        if metric in overall:
            print(f"{metric}={overall[metric]}")

    _print_counter("Issue Buckets", summary["issue_counts"], limit)
    _print_counter("GT Tool", summary["by_gt_tool"], limit)
    _print_counter("Pred Tool/Type", summary["by_pred_tool"], limit)
    _print_counter("Files", summary["by_file"], limit)
    _print_counter("Categories", summary["by_category"], limit)
    _print_counter("Arg Slot Deltas", summary.get("slot_counts", {}), limit)
    _print_counter("Confusions", summary.get("confusion_counts", {}), limit)

    print("\n## Examples")
    for issue, examples in list(summary["examples"].items())[:limit]:
        print(f"\n### {issue}")
        for item in examples[:3]:
            print(f"- {item['file']}:{item['id']} {item['query']}")
            print(f"  gt={item['gt_tool']} {item['gt_args']}")
            print(f"  pred={item['pred_type']} {item['pred_tool']} {item['pred_args']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to eval_report*.json")
    parser.add_argument("--tools", default=str(ROOT / "data" / "tools.json"), help="Tool schema JSON path")
    parser.add_argument("--limit", type=int, default=15, help="Rows per summary section")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--backlog-md", help="Write a Markdown training backlog to this path")
    args = parser.parse_args()

    summary = summarize_report(load_report(args.report), args.tools)
    if args.backlog_md:
        backlog = build_training_backlog(summary, args.limit)
        out_path = Path(args.backlog_md)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_training_backlog_markdown(backlog), encoding="utf-8")
        print(f"[backlog] wrote {len(backlog)} items -> {out_path}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_summary(summary, args.limit)


if __name__ == "__main__":
    main()
