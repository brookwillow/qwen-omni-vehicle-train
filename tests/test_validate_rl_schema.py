from scripts.validate_rl_schema import iter_tool_calls, validate_tool_call


SCHEMA = {
    "WindowControl": {
        "props": {
            "action": {"type": "string", "enum": ["打开", "关闭", "开到"]},
            "device": {"type": "string", "enum": ["车窗"]},
            "position": {"type": "string", "enum": ["前排"]},
            "value": {"type": "string", "enum": ["通风"]},
        },
        "required": ["action", "device"],
    },
    "NoiseDoNotAct": {"props": {}, "required": []},
}


def test_iter_tool_calls_finds_nested_preference_tool_calls():
    row = {
        "chosen": '{"name":"WindowControl","arguments":{"action":"关闭","device":"车窗"}}',
        "expected": {
            "target_tool_call": {
                "name": "WindowControl",
                "arguments": {"action": "打开", "device": "车窗"},
            }
        },
    }

    calls = iter_tool_calls(row, "$")

    assert [call[0] for call in calls] == ["WindowControl", "WindowControl"]


def test_validate_tool_call_reports_invalid_enum():
    issues = validate_tool_call(
        "WindowControl",
        {"action": "关闭", "device": "前窗"},
        "$.rejected",
        SCHEMA,
    )

    assert issues[0]["type"] == "INVALID_ENUM"
    assert issues[0]["param"] == "device"


def test_validate_tool_call_allows_window_percentage_value():
    issues = validate_tool_call(
        "WindowControl",
        {"action": "开到", "device": "车窗", "value": "50%"},
        "$.chosen",
        SCHEMA,
    )

    assert issues == []


def test_validate_tool_call_allows_numeric_value_with_description():
    schema = {
        "VoiceControl": {
            "props": {
                "action": {"type": "string", "enum": ["调到"]},
                "feature": {"type": "string", "enum": ["声音"]},
                "value": {
                    "type": "string",
                    "enum": ["最高", "最低"],
                    "description": "numeric and percentage values alongside enums",
                },
            },
            "required": ["action", "feature"],
        }
    }

    issues = validate_tool_call(
        "VoiceControl",
        {"action": "调到", "feature": "声音", "value": "30"},
        "$.chosen",
        schema,
    )

    assert issues == []


def test_anti_over_noise_preferences_cover_reviewed_false_positives():
    import json
    from pathlib import Path

    path = Path("data/rl/anti_over_noise_preferences.jsonl")
    assert path.exists()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_by_source = {row["source_eval_key"]: row for row in rows}
    reviewed_sources = {
        "black_box_test.json#black_box_009",
        "black_box_test.json#black_box_110",
        "seat_test.json#seat_045",
        "seat_test.json#seat_049",
        "seat_test.json#seat_050",
        "seat_test.json#seat_051",
        "seat_test.json#seat_056",
        "steering_test.json#steering_014",
        "steering_test.json#steering_018",
        "steering_test.json#steering_045",
        "voice_test.json#voice_010",
        "voice_test.json#voice_011",
        "voice_test.json#voice_031",
        "voice_test.json#voice_032",
        "voice_test.json#voice_035",
        "voice_test.json#voice_036",
        "voice_test.json#voice_043",
        "voice_test.json#voice_044",
        "window_test.json#window_011",
        "window_test.json#window_015",
        "window_test.json#window_030",
        "window_test.json#window_040",
        "window_test.json#window_041",
        "window_test.json#window_042",
    }

    assert len(rows) == 70
    assert reviewed_sources.issubset(rows_by_source)
    assert {row["task_type"] for row in rows} == {"valid_tool_vs_noise_false_positive"}
    assert {row["rejected"]["name"] for row in rows} == {"NoiseDoNotAct"}
    assert all(row["rejected"]["arguments"] == {} for row in rows)
    assert all(row["chosen"]["name"] != "NoiseDoNotAct" for row in rows)
    assert {
        "VoiceControl",
        "SeatControl",
        "WindowControl",
        "SteeringwheelControl",
    }.issubset({row["chosen"]["name"] for row in rows})


def test_eval_error_dpo_preferences_cover_suitable_error_types():
    import json
    from pathlib import Path

    expected = {
        "still_over_noise_preferences_round2.jsonl": ("still_over_noise", 49),
        "wrong_tool_preferences.jsonl": ("wrong_tool", 49),
        "false_reject_clarify_preferences.jsonl": ("false_reject_or_clarify", 12),
        "extra_args_preferences.jsonl": ("extra_args", 90),
    }

    for filename, (task_type, count) in expected.items():
        path = Path("data/rl") / filename
        assert path.exists(), filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == count, filename
        assert {row["task_type"] for row in rows} == {task_type}
        assert all(row["chosen"]["name"] != "NoiseDoNotAct" for row in rows)
        assert all(row["source_eval_key"] for row in rows)

    still_noise_rows = [
        json.loads(line)
        for line in Path("data/rl/still_over_noise_preferences_round2.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["rejected"]["name"] for row in still_noise_rows} == {"NoiseDoNotAct"}
    assert all(row["rejected"]["arguments"] == {} for row in still_noise_rows)

    wrong_tool_rows = [
        json.loads(line)
        for line in Path("data/rl/wrong_tool_preferences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["chosen"]["name"] != row["rejected"]["name"] for row in wrong_tool_rows)

    extra_arg_rows = [
        json.loads(line)
        for line in Path("data/rl/extra_args_preferences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(row["chosen"]["name"] == row["rejected"]["name"] for row in extra_arg_rows)
    assert all(
        set(row["rejected"]["arguments"]) - set(row["chosen"]["arguments"])
        for row in extra_arg_rows
    )
