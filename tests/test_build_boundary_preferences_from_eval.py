import json

from scripts.build_boundary_preferences_from_eval import build_boundary_preferences


def test_build_boundary_preferences_from_eval_report(tmp_path):
    report = {
        "errors": [
            {
                "id": "a",
                "file": "window_test.json",
                "err_type": "tool-err",
                "query": "打开车窗",
                "expected_type": "Action",
                "gt_tool": "WindowControl",
                "gt_args": {"action": "打开", "device": "车窗"},
                "pred_tool": "NoiseDoNotAct",
            },
            {
                "id": "b",
                "file": "camera_test.json",
                "err_type": "type-err",
                "query": "打开右侧摄像头",
                "expected_type": "Action",
                "gt_tool": "CameraControl",
                "gt_args": {"action": "打开", "device": "摄像头", "position": "右侧"},
                "pred_type": "Reject",
            },
            {
                "id": "c",
                "file": "noise_test.json",
                "err_type": "args-err",
                "query": "忽略",
                "expected_type": "Action",
                "gt_tool": "WindowControl",
                "gt_args": {},
                "pred_tool": "WindowControl",
            },
        ]
    }
    report_path = tmp_path / "eval_report_20260609_105416.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    over_path, reject_path, over_count, reject_count = build_boundary_preferences(report_path, tmp_path)

    assert over_count == 1
    assert reject_count == 1

    over_row = json.loads(over_path.read_text(encoding="utf-8").strip())
    reject_row = json.loads(reject_path.read_text(encoding="utf-8").strip())

    assert over_row["chosen"]["name"] == "WindowControl"
    assert over_row["rejected"]["name"] == "NoiseDoNotAct"
    assert reject_row["chosen"]["name"] == "CameraControl"
    assert reject_row["rejected"] == "Reject。"
