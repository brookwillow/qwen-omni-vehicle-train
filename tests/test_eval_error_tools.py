import json

from scripts.analyze_eval_errors import build_training_backlog, classify_error, summarize_report


def test_classifies_missing_feature_as_missing_optional_arg(tmp_path):
    tools = tmp_path / "tools.json"
    tools.write_text(
        json.dumps(
            [
                {
                    "name": "ScreenControl",
                    "inputSchema": {"required": ["action", "device"]},
                }
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "overall": {"total": 1},
        "errors": [
            {
                "err_type": "args-err",
                "gt_tool": "ScreenControl",
                "gt_args": {"action": "调到", "device": "仪表", "feature": "亮度", "value": "最高"},
                "pred_args": {"action": "调到", "device": "仪表", "value": "最高"},
            }
        ],
    }

    summary = summarize_report(report, tools)

    assert summary["issue_counts"] == {"missing_optional_args:feature": 1}


def test_classifies_over_reject():
    issue = classify_error(
        {
            "err_type": "type-err",
            "expected_type": "Action",
            "pred_type": "Reject",
        }
    )

    assert issue == "over_reject"


def test_build_training_backlog_prioritizes_arg_errors():
    summary = {
        "issue_counts": {
            "wrong_arg_value:action": 42,
            "over_noise": 59,
        },
        "by_gt_tool": {
            "ClimateControl": 107,
            "WindowControl": 69,
        },
        "examples": {
            "wrong_arg_value:action": [
                {
                    "query": "前排车窗下降",
                    "gt_tool": "WindowControl",
                    "gt_args": {"action": "再开", "device": "车窗", "position": "前排"},
                    "pred_tool": "WindowControl",
                    "pred_args": {"action": "打开", "device": "车窗", "position": "前排"},
                }
            ]
        },
    }

    backlog = build_training_backlog(summary, limit=2)

    assert backlog[0]["issue"] == "wrong_arg_value:action"
    assert backlog[0]["priority"] == "P0"
    assert backlog[0]["training_focus"] == "参数值强对比"
