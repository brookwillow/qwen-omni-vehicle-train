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
