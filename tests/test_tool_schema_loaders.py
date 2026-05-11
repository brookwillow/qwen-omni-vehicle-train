import json

from infer_cli_omni import load_tools, validate_action
from scripts.validate_splits import load_schema, validate_sample


def test_validate_splits_load_schema_supports_input_schema(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            [
                {
                    "name": "ExampleControl",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["打开"]},
                            "device": {"type": "string", "enum": ["示例"]},
                        },
                        "required": ["action", "device"],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = load_schema(str(tools_path))
    sample = {
        "messages": [
            {
                "role": "assistant",
                "content": '{"name":"ExampleControl","arguments":{"action":"打开","device":"示例"}}',
            }
        ]
    }

    assert schema["ExampleControl"]["required"] == ["action", "device"]
    assert validate_sample(sample, "sample", schema) == []


def test_infer_cli_load_tools_supports_input_schema(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            [
                {
                    "name": "ExampleControl",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"action": {"type": "string", "enum": ["打开"]}},
                        "required": ["action"],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tool_map = load_tools(str(tools_path))

    assert validate_action(tool_map, "ExampleControl", {"action": "打开"})
    assert not validate_action(tool_map, "ExampleControl", {"action": "关闭"})


def test_value_parameters_allow_numeric_strings_outside_enum(tmp_path):
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(
        json.dumps(
            [
                {
                    "name": "ExampleControl",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["调到"]},
                            "value": {"type": "string", "enum": ["最高", "最低"]},
                        },
                        "required": ["action"],
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sample = {
        "messages": [
            {
                "role": "assistant",
                "content": '{"name":"ExampleControl","arguments":{"action":"调到","value":"20.5"}}',
            }
        ]
    }

    schema = load_schema(str(tools_path))
    tool_map = load_tools(str(tools_path))

    assert validate_sample(sample, "sample", schema) == []
    assert validate_action(tool_map, "ExampleControl", {"action": "调到", "value": "50%"})
