import json
from pathlib import Path

from scripts.schema_coverage_report import build_coverage, build_hard_case_backlog, iter_eval_calls, iter_sft_calls


SCHEMA = {
    "WindowControl": {
        "props": {
            "action": {"type": "string", "enum": ["打开", "关闭"]},
            "device": {"type": "string", "enum": ["车窗", "天窗"]},
            "position": {"type": "string", "enum": ["主驾", "副驾"]},
        },
        "required": ["action", "device"],
    }
}


def test_schema_coverage_finds_eval_combo_missing_in_sft(tmp_path: Path):
    splits = tmp_path / "splits"
    eval_dir = tmp_path / "eval"
    splits.mkdir()
    eval_dir.mkdir()

    (splits / "WindowControl.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "打开车窗"},
                    {
                        "role": "assistant",
                        "content": '{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}',
                    },
                ]
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "window_test.json").write_text(
        json.dumps(
            [
                {
                    "id": "window_001",
                    "query": "关闭天窗",
                    "expected_tool_calls": [
                        {"name": "WindowControl", "arguments": {"action": "关闭", "device": "天窗"}}
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = iter_sft_calls(splits) + iter_eval_calls(eval_dir)
    report = build_coverage(calls, SCHEMA)

    assert report["source_tool_counts"]["sft"]["WindowControl"] == 1
    assert report["source_tool_counts"]["eval"]["WindowControl"] == 1
    assert report["eval_combos_missing_in_sft"][0]["args"] == {"action": "关闭", "device": "天窗"}
    assert any(row["tool"] == "WindowControl" and row["param"] == "position" for row in report["missing_enum_values"]["sft"])


def test_schema_coverage_keeps_source_specific_examples():
    calls = [
        ("sft", "WindowControl", {"action": "打开", "device": "车窗"}, "sft:1"),
        ("eval", "WindowControl", {"action": "关闭", "device": "天窗"}, "eval:case"),
        ("rl", "WindowControl", {"action": "关闭", "device": "天窗"}, "rl:pair"),
    ]

    report = build_coverage(calls, SCHEMA)

    assert report["eval_combos_missing_in_sft"][0]["example"] == "eval:case"
    assert report["rl_combos_missing_in_sft"][0]["example"] == "rl:pair"


def test_coverage_backlog_groups_missing_combos_by_tool():
    report = {
        "eval_combos_missing_in_sft": [
            {
                "tool": "WindowControl",
                "args": {"action": "关闭", "device": "天窗"},
                "eval_count": 2,
                "example": "data/eval/window_test.json:window_001",
            },
            {
                "tool": "WindowControl",
                "args": {"action": "打开", "device": "天窗"},
                "eval_count": 1,
                "example": "data/eval/window_test.json:window_002",
            },
        ],
        "rl_combos_missing_in_sft": [],
        "missing_enum_values": {
            "sft": [
                {
                    "tool": "WindowControl",
                    "param": "position",
                    "seen": 0,
                    "total": 2,
                    "missing": ["主驾", "副驾"],
                }
            ]
        },
    }

    backlog = build_hard_case_backlog(report)

    assert backlog[0]["priority"] == "P1"
    assert backlog[0]["issue"] == "eval_missing_combo:WindowControl"
    assert len(backlog[0]["items"]) == 2
    assert any(item["issue"] == "sft_missing_enum:WindowControl.position" for item in backlog)
