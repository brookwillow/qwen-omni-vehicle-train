#!/usr/bin/env python3
"""Cluster eval report errors into actionable buckets."""

from __future__ import annotations

import argparse
import html
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


def _review_error(error: dict[str, Any], required_by_tool: dict[str, set[str]]) -> dict[str, Any]:
    gt_args = error.get("gt_args") or {}
    pred_args = error.get("pred_args") or {}
    return {
        **error,
        "issue": classify_error(error, required_by_tool),
        "arg_delta": arg_delta(gt_args, pred_args)
        if isinstance(gt_args, dict) and isinstance(pred_args, dict)
        else {},
    }


def format_review_html(report: dict[str, Any], tools_path: str | Path, source_path: str | Path) -> str:
    """Return a self-contained HTML review page for every eval error."""
    required_by_tool = _required_fields_by_tool(tools_path)
    errors = [_review_error(error, required_by_tool) for error in report.get("errors", []) or []]
    summary = summarize_report(report, tools_path)
    payload = {
        "source": str(source_path),
        "timestamp": report.get("timestamp"),
        "lora_dir": report.get("lora_dir"),
        "overall": report.get("overall", {}),
        "errors": errors,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    issue_counts = summary.get("issue_counts", {})
    by_file = summary.get("by_file", {})
    issue_items = "".join(
        f"<li><code>{html.escape(issue)}</code><span>{count}</span></li>"
        for issue, count in list(issue_counts.items())[:12]
    )
    file_options = "\n".join(
        f'<option value="{html.escape(name)}">{html.escape(name)} ({count})</option>'
        for name, count in by_file.items()
    )
    title = f"SFT Eval Error Review - {Path(source_path).name}"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #16202a;
      --muted: #627386;
      --accent: #1769aa;
      --bad: #a33b20;
      --ok: #176f45;
      --warn: #8a5b00;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }}
    header {{ position: sticky; top: 0; z-index: 3; background: rgba(246,247,249,.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(8px); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 20px 24px; }}
    h1 {{ font-size: 24px; margin: 0 0 10px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    .meta, .muted {{ color: var(--muted); font-size: 13px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-top: 14px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
    .metric strong {{ display: block; font-size: 20px; }}
    .toolbar {{ display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr auto; gap: 10px; margin-top: 14px; }}
    input, select, button, textarea {{ font: inherit; }}
    input, select {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; background: #fff; }}
    button {{ border: 1px solid var(--accent); background: var(--accent); color: #fff; border-radius: 6px; padding: 9px 14px; cursor: pointer; white-space: nowrap; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 22px 24px 60px; }}
    .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
    .summary-card, .case {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .summary-card {{ padding: 16px; }}
    .summary-card ul {{ margin: 0; padding: 0; list-style: none; columns: 2; }}
    .summary-card li {{ display: flex; justify-content: space-between; gap: 8px; break-inside: avoid; margin-bottom: 7px; }}
    .case {{ margin-bottom: 14px; overflow: hidden; }}
    .case-head {{ display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: center; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .case-id {{ font-weight: 700; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tag {{ border: 1px solid var(--line); background: #f9fafb; color: var(--muted); border-radius: 999px; padding: 3px 8px; font-size: 12px; }}
    .err-type {{ color: var(--bad); border-color: #efc2b4; background: #fff3ef; }}
    .issue {{ color: var(--warn); border-color: #e6c777; background: #fff8df; }}
    .case-body {{ padding: 14px 16px 16px; }}
    .query {{ font-size: 17px; font-weight: 650; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .box {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfcfe; min-width: 0; }}
    .box h3 {{ margin: 0 0 8px; font-size: 14px; color: var(--muted); }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.45; }}
    .review {{ display: grid; grid-template-columns: repeat(3, minmax(180px, auto)) 1fr; gap: 10px; align-items: center; margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }}
    .review label {{ display: flex; align-items: center; gap: 6px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: #fff; }}
    .review input[type="radio"] {{ width: auto; }}
    textarea {{ width: 100%; min-height: 38px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      .toolbar, .summary, .grid, .review, .metrics {{ grid-template-columns: 1fr; }}
      .case-head {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{html.escape(title)}</h1>
      <div class="meta">source: <code>{html.escape(str(source_path))}</code> · lora: <code>{html.escape(str(report.get("lora_dir") or "-"))}</code> · timestamp: {html.escape(str(report.get("timestamp") or "-"))}</div>
      <div class="metrics" id="metrics"></div>
      <div class="toolbar">
        <input id="q" type="search" placeholder="搜索 query / id / 工具 / 参数">
        <select id="file"><option value="">全部文件</option>{file_options}</select>
        <select id="err"><option value="">全部错误类型</option><option>type-err</option><option>tool-err</option><option>args-err</option></select>
        <select id="decision"><option value="">全部判定</option><option value="unreviewed">未判定</option><option value="fix_eval">修改 eval</option><option value="model_fail">预测失败</option><option value="needs_discussion">待确认</option></select>
        <button id="export">导出判定 JSON</button>
      </div>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="summary-card">
        <h2>Top Issue Buckets</h2>
        <ul>{issue_items}</ul>
      </div>
      <div class="summary-card">
        <h2>使用方式</h2>
        <p class="muted">逐条选择“修改 eval”“预测失败”或“待确认”。选择和备注会自动保存在浏览器 localStorage；点击导出可得到后续清洗/训练回流用的 JSON。</p>
      </div>
    </section>
    <div class="meta" id="count"></div>
    <section id="cases"></section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const storageKey = 'eval-review:' + payload.source + ':' + (payload.timestamp || '');
    const reviews = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
    const fields = [
      ['total', payload.overall.total],
      ['type_acc', payload.overall.type_acc],
      ['tool_acc', payload.overall.tool_acc],
      ['args_em', payload.overall.args_em],
      ['errors', payload.errors.length],
      ['reviewed', Object.keys(reviews).filter(k => reviews[k]?.decision).length],
    ];
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const pretty = (value) => esc(JSON.stringify(value ?? null, null, 2));
    const caseKey = (item) => `${{item.file}}#${{item.id}}`;
    const saveReview = (key, patch) => {{
      reviews[key] = {{...(reviews[key] || {{}}), ...patch}};
      localStorage.setItem(storageKey, JSON.stringify(reviews));
      renderMetrics();
    }};
    function renderMetrics() {{
      fields[5][1] = Object.keys(reviews).filter(k => reviews[k]?.decision).length;
      document.getElementById('metrics').innerHTML = fields.map(([k,v]) => `<div class="metric"><span class="muted">${{esc(k)}}</span><strong>${{esc(v)}}</strong></div>`).join('');
    }}
    function render() {{
      const q = document.getElementById('q').value.trim().toLowerCase();
      const file = document.getElementById('file').value;
      const err = document.getElementById('err').value;
      const decision = document.getElementById('decision').value;
      const items = payload.errors.filter(item => {{
        const key = caseKey(item);
        const rv = reviews[key] || {{}};
        if (file && item.file !== file) return false;
        if (err && item.err_type !== err) return false;
        if (decision === 'unreviewed' && rv.decision) return false;
        if (decision && decision !== 'unreviewed' && rv.decision !== decision) return false;
        if (q) {{
          const hay = JSON.stringify(item).toLowerCase();
          if (!hay.includes(q)) return false;
        }}
        return true;
      }});
      document.getElementById('count').textContent = `显示 ${{items.length}} / ${{payload.errors.length}} 条`;
      document.getElementById('cases').innerHTML = items.map((item, index) => {{
        const key = caseKey(item);
        const rv = reviews[key] || {{}};
        const decisionValue = rv.decision || '';
        const radio = (value, label) => `<label><input type="radio" name="decision-${{esc(key)}}\" value="${{value}}" ${{decisionValue === value ? 'checked' : ''}}> ${{label}}</label>`;
        return `<article class="case" data-key="${{esc(key)}}">
          <div class="case-head">
            <div class="case-id">${{index + 1}}. ${{esc(key)}}</div>
            <div class="tags">
              <span class="tag err-type">${{esc(item.err_type)}}</span>
              <span class="tag issue">${{esc(item.issue)}}</span>
              <span class="tag">${{esc(item.category)}}</span>
              <span class="tag">${{esc(item.difficulty)}}</span>
            </div>
            <div class="muted">${{esc(item.file)}}</div>
          </div>
          <div class="case-body">
            <div class="query">${{esc(item.query)}}</div>
            <div class="grid">
              <div class="box"><h3>Expected</h3><pre>${{pretty({{type:item.expected_type, tool:item.gt_tool, args:item.gt_args, tool_calls:item.expected_tool_calls}})}}</pre></div>
              <div class="box"><h3>Predicted</h3><pre>${{pretty({{type:item.pred_type, tool:item.pred_tool, args:item.pred_args, tool_calls:item.pred_tool_calls, raw:item.pred_raw}})}}</pre></div>
              <div class="box"><h3>Arg Delta</h3><pre>${{pretty(item.arg_delta)}}</pre></div>
              <div class="box"><h3>Review Key</h3><pre>${{esc(key)}}</pre></div>
            </div>
            <div class="review">
              ${{radio('fix_eval', '修改 eval')}}
              ${{radio('model_fail', '预测失败')}}
              ${{radio('needs_discussion', '待确认')}}
              <textarea placeholder="备注，例如：GT 应改成 CameraControl / feature 应为音乐应用 / 模型漏 position" data-note="${{esc(key)}}">${{esc(rv.note || '')}}</textarea>
            </div>
          </div>
        </article>`;
      }}).join('');
      document.querySelectorAll('.case input[type="radio"]').forEach(input => {{
        input.addEventListener('change', event => {{
          const article = event.target.closest('.case');
          saveReview(article.dataset.key, {{decision: event.target.value}});
        }});
      }});
      document.querySelectorAll('textarea[data-note]').forEach(input => {{
        input.addEventListener('input', event => saveReview(event.target.dataset.note, {{note: event.target.value}}));
      }});
    }}
    for (const id of ['q', 'file', 'err', 'decision']) document.getElementById(id).addEventListener('input', render);
    document.getElementById('export').addEventListener('click', () => {{
      const rows = payload.errors.map(item => {{
        const key = caseKey(item);
        return {{
          key,
          id: item.id,
          file: item.file,
          query: item.query,
          err_type: item.err_type,
          issue: item.issue,
          gt_tool: item.gt_tool,
          gt_args: item.gt_args,
          pred_type: item.pred_type,
          pred_tool: item.pred_tool,
          pred_args: item.pred_args,
          decision: reviews[key]?.decision || '',
          note: reviews[key]?.note || '',
        }};
      }});
      const blob = new Blob([JSON.stringify(rows, null, 2)], {{type: 'application/json;charset=utf-8'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'eval_error_review_' + (payload.timestamp || 'report').replace(/[:.]/g, '-') + '.json';
      a.click();
      URL.revokeObjectURL(a.href);
    }});
    renderMetrics();
    render();
  </script>
</body>
</html>
"""


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
    parser.add_argument("--review-html", help="Write a self-contained HTML page for manual error review")
    args = parser.parse_args()

    summary = summarize_report(load_report(args.report), args.tools)
    if args.review_html:
        report = load_report(args.report)
        out_path = Path(args.review_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(format_review_html(report, args.tools, args.report), encoding="utf-8")
        print(f"[review] wrote {len(report.get('errors', []) or [])} cases -> {out_path}")
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
