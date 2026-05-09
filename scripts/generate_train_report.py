#!/usr/bin/env python3
"""
Generate an HTML training report from train_metrics.jsonl.

Usage:
  python3 scripts/generate_train_report.py --metrics lora_output/train_metrics.jsonl
  python3 scripts/generate_train_report.py --metrics lora_output/train_metrics.jsonl --out report.html
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

# ── HTML template (Chart.js via CDN, no extra Python deps) ────

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>训练报告 · {run_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2d3148;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6366f1;
    --green: #22d3a0; --yellow: #f59e0b; --red: #f87171; --blue: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 28px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 28px; }}
  .stat {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 20px; }}
  .stat .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }}
  .stat .value {{ font-size: 1.4rem; font-weight: 700; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 20px; }}
  .chart-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .chart-card h2 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
  canvas {{ max-height: 240px; }}
  .footer {{ margin-top: 28px; text-align: center; color: var(--muted); font-size: 0.75rem; }}
</style>
</head>
<body>

<h1>训练报告</h1>
<div class="subtitle">运行: {run_name} &nbsp;·&nbsp; 生成时间: {generated_at}</div>

<div class="grid">
  <div class="stat"><div class="label">总步数</div><div class="value" style="color:var(--accent)">{total_steps}</div></div>
  <div class="stat"><div class="label">训练 epoch</div><div class="value" style="color:var(--blue)">{total_epochs}</div></div>
  <div class="stat"><div class="label">最终 train loss</div><div class="value" style="color:var(--yellow)">{final_train_loss}</div></div>
  <div class="stat"><div class="label">最优 eval loss</div><div class="value" style="color:var(--green)">{best_eval_loss}</div></div>
  <div class="stat"><div class="label">最优 eval acc</div><div class="value" style="color:var(--green)">{best_eval_acc}</div></div>
</div>

<div class="charts">

  <div class="chart-card">
    <h2>Train Loss</h2>
    <canvas id="trainLoss"></canvas>
  </div>

  <div class="chart-card">
    <h2>Eval Loss &amp; Eval Token Acc</h2>
    <canvas id="evalMetrics"></canvas>
  </div>

  <div class="chart-card">
    <h2>Learning Rate</h2>
    <canvas id="lr"></canvas>
  </div>

  <div class="chart-card">
    <h2>Grad Norm</h2>
    <canvas id="gradNorm"></canvas>
  </div>

</div>

<div class="footer">由 scripts/generate_train_report.py 自动生成 · {run_name}</div>

<script>
const TRAIN = {train_json};
const EVAL  = {eval_json};

function makeChart(id, datasets, yLabel, opts) {{
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {{
    type: 'line',
    data: {{ datasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#8892a4', boxWidth: 12, font: {{ size: 11 }} }} }},
        tooltip: {{ bodyFont: {{ size: 11 }}, titleFont: {{ size: 11 }} }},
      }},
      scales: {{
        x: {{
          type: 'linear', title: {{ display: true, text: 'Step', color: '#8892a4', font: {{ size: 11 }} }},
          ticks: {{ color: '#8892a4', font: {{ size: 10 }} }},
          grid: {{ color: '#2d3148' }},
        }},
        y: {{
          title: {{ display: true, text: yLabel, color: '#8892a4', font: {{ size: 11 }} }},
          ticks: {{ color: '#8892a4', font: {{ size: 10 }} }},
          grid: {{ color: '#2d3148' }},
          ...opts,
        }},
      }},
    }},
  }});
}}

// Train loss
makeChart('trainLoss', [{{
  label: 'train loss',
  data: TRAIN.map(r => ({{ x: r.step, y: r.loss }})).filter(r => r.y != null),
  borderColor: '#f59e0b', backgroundColor: '#f59e0b22',
  pointRadius: 0, borderWidth: 1.5, fill: true, tension: 0.3,
}}], 'Loss', {{}});

// Eval loss + acc (dual axis)
const evalLossDs = {{
  label: 'eval loss',
  data: EVAL.map(r => ({{ x: r.step, y: r.eval_loss }})).filter(r => r.y != null),
  borderColor: '#6366f1', backgroundColor: '#6366f122',
  pointRadius: 3, borderWidth: 2, fill: false, yAxisID: 'y',
}};
const evalAccDs = {{
  label: 'eval token acc',
  data: EVAL.map(r => ({{ x: r.step, y: r.eval_token_acc }})).filter(r => r.y != null),
  borderColor: '#22d3a0', backgroundColor: '#22d3a022',
  pointRadius: 3, borderWidth: 2, fill: false, yAxisID: 'y2',
}};
(function() {{
  const ctx = document.getElementById('evalMetrics').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{ datasets: [evalLossDs, evalAccDs] }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#8892a4', boxWidth: 12, font: {{ size: 11 }} }} }},
      }},
      scales: {{
        x: {{
          type: 'linear', title: {{ display: true, text: 'Step', color: '#8892a4', font: {{ size: 11 }} }},
          ticks: {{ color: '#8892a4', font: {{ size: 10 }} }}, grid: {{ color: '#2d3148' }},
        }},
        y: {{
          position: 'left',
          title: {{ display: true, text: 'Eval Loss', color: '#6366f1', font: {{ size: 11 }} }},
          ticks: {{ color: '#8892a4', font: {{ size: 10 }} }}, grid: {{ color: '#2d3148' }},
        }},
        y2: {{
          position: 'right',
          title: {{ display: true, text: 'Token Acc', color: '#22d3a0', font: {{ size: 11 }} }},
          ticks: {{ color: '#8892a4', font: {{ size: 10 }} }}, grid: {{ drawOnChartArea: false }},
          min: 0, max: 1,
        }},
      }},
    }},
  }});
}})();

// Learning rate
makeChart('lr', [{{
  label: 'learning rate',
  data: TRAIN.map(r => ({{ x: r.step, y: r.learning_rate }})).filter(r => r.y != null),
  borderColor: '#38bdf8', backgroundColor: '#38bdf822',
  pointRadius: 0, borderWidth: 1.5, fill: true, tension: 0.3,
}}], 'LR', {{}});

// Grad norm
makeChart('gradNorm', [{{
  label: 'grad norm',
  data: TRAIN.map(r => ({{ x: r.step, y: r.grad_norm }})).filter(r => r.y != null),
  borderColor: '#f87171', backgroundColor: '#f8717122',
  pointRadius: 0, borderWidth: 1.5, fill: true, tension: 0.3,
}}], 'Grad Norm', {{}});
</script>
</body>
</html>
"""


def fmt(v: float | None, precision: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{precision}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Path to train_metrics.jsonl")
    parser.add_argument("--out", default=None, help="Output HTML path (default: same dir as metrics)")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Not found: {metrics_path}")

    records = [json.loads(l) for l in metrics_path.open(encoding="utf-8") if l.strip()]
    if not records:
        raise ValueError("train_metrics.jsonl is empty.")

    train_records = [r for r in records if "loss" in r]
    eval_records  = [r for r in records if "eval_loss" in r]

    # Summary stats
    total_steps  = max((r["step"] for r in records), default=0)
    total_epochs = max((r["epoch"] for r in records if r.get("epoch") is not None), default=0)
    final_train_loss = train_records[-1]["loss"] if train_records else None
    best_eval_loss   = min((r["eval_loss"] for r in eval_records), default=None)
    best_eval_acc    = max((r.get("eval_token_acc", 0) for r in eval_records), default=None)

    run_name     = metrics_path.parent.name
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = Path(args.out) if args.out else metrics_path.parent / "train_report.html"

    html = HTML_TEMPLATE.format(
        run_name=run_name,
        generated_at=generated_at,
        total_steps=total_steps,
        total_epochs=fmt(total_epochs, 1),
        final_train_loss=fmt(final_train_loss),
        best_eval_loss=fmt(best_eval_loss),
        best_eval_acc=fmt(best_eval_acc),
        train_json=json.dumps(train_records, ensure_ascii=False),
        eval_json=json.dumps(eval_records, ensure_ascii=False),
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Report saved → {out_path}")


if __name__ == "__main__":
    main()
