import json
import random
from pathlib import Path

from build_train_data import (
    load_weighted_splits,
    match_weight_selector,
    parse_kv_args,
    validate_sample_schema,
)
from scripts.validate_splits import validate_sample


def _write_jsonl(path: Path, queries: list[str]) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "{\"name\":\"WindowControl\",\"arguments\":{\"action\":\"打开\",\"device\":\"车窗\"}}"},
            ]
        }
        for query in queries
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_match_weight_selector_accepts_stem_path_and_glob():
    path = Path("data/splits/hard_cases/window_args.jsonl")

    assert match_weight_selector(path, "window_args")
    assert match_weight_selector(path, "hard_cases/window_args.jsonl")
    assert match_weight_selector(path, "hard_cases/*.jsonl")
    assert not match_weight_selector(path, "ClimateControl")


def test_load_weighted_splits_oversamples_matching_file(tmp_path):
    normal = tmp_path / "WindowControl.jsonl"
    hard_dir = tmp_path / "hard_cases"
    hard_dir.mkdir()
    hard = hard_dir / "window_args.jsonl"
    _write_jsonl(normal, ["普通开窗"])
    _write_jsonl(hard, ["强对比开窗"])

    samples, counts = load_weighted_splits(
        [normal, hard],
        oversample={},
        max_per_type={},
        sample_weights={"hard_cases/*.jsonl": 3.0},
        rng=random.Random(7),
    )

    queries = [sample["messages"][0]["content"] for sample in samples]
    assert queries.count("普通开窗") == 1
    assert queries.count("强对比开窗") == 3
    assert counts[str(normal)] == 1
    assert counts[str(hard)] == 3


def test_parse_kv_args_preserves_string_selectors():
    assert parse_kv_args(["hard_cases/*.jsonl:2.5"]) == {"hard_cases/*.jsonl": 2.5}


# ── Schema validation tests ──────────────────────────────────

_SCHEMA = {
    "WindowControl": {
        "props": {
            "action": {"type": "string", "enum": ["关闭", "打开", "开到", "关到", "再开", "再关", "暂停"]},
            "device": {"type": "string", "enum": ["车窗", "遮阳帘", "顶遮阳帘", "侧遮阳帘", "天幕", "天窗"]},
            "position": {"type": "string", "enum": ["主驾", "副驾", "全部"]},
            "value": {"type": "string", "enum": ["通风"], "description": "numeric and percentage values alongside enums"},
        },
        "required": ["action", "device"],
    },
    "ClimateControl": {
        "props": {
            "action": {"type": "string", "enum": ["关闭", "打开", "调到", "调高", "调低"]},
            "device": {"type": "string", "enum": ["空调", "空气净化器"]},
            "feature": {"type": "string", "enum": ["温度", "除雾", "风"]},
            "value": {"type": "string", "enum": ["制冷", "制热", "中"]},
        },
        "required": ["action"],
    },
    "NoiseDoNotAct": {
        "props": {},
        "required": [],
    },
}


def _make_sample(tool_call_json: str) -> dict:
    return {"messages": [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": tool_call_json},
    ]}


def test_validate_schema_valid_sample():
    sample = _make_sample('{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}')
    assert validate_sample_schema(sample, _SCHEMA) == []


def test_validate_schema_missing_required():
    sample = _make_sample('{"name":"WindowControl","arguments":{"action":"打开"}}')
    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("missing required field" in e for e in errs)


def test_validate_schema_invalid_enum():
    sample = _make_sample('{"name":"WindowControl","arguments":{"action":"飞行","device":"车窗"}}')
    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("invalid enum" in e for e in errs)


def test_validate_schema_unknown_tool():
    sample = _make_sample('{"name":"FlyingControl","arguments":{"action":"fly"}}')
    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("unknown tool" in e for e in errs)


def test_validate_schema_numeric_value_allowed():
    sample = _make_sample('{"name":"WindowControl","arguments":{"action":"开到","device":"车窗","value":"50%"}}')
    assert validate_sample_schema(sample, _SCHEMA) == []


def test_validate_schema_noise_always_valid():
    sample = _make_sample('{"name":"NoiseDoNotAct","arguments":{}}')
    assert validate_sample_schema(sample, _SCHEMA) == []


def test_validate_schema_unknown_field():
    sample = _make_sample('{"name":"WindowControl","arguments":{"action":"打开","device":"车窗","color":"red"}}')
    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("unknown field" in e for e in errs)


def test_validate_schema_accepts_multi_tool_array():
    sample = _make_sample(
        '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
        '{"name":"ClimateControl","arguments":{"action":"打开","device":"空调"}}]'
    )

    assert validate_sample_schema(sample, _SCHEMA) == []


def test_validate_schema_reports_errors_inside_multi_tool_array():
    sample = _make_sample(
        '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
        '{"name":"ClimateControl","arguments":{"action":"飞行","device":"空调"}}]'
    )

    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("ClimateControl.action: invalid enum" in e for e in errs)


def test_split_validator_ignores_popup_option_arrays():
    sample = {
        "messages": [
            {
                "role": "assistant",
                "content": '[{"index": 1, "name": "舒适"}, {"index": 2, "name": "运动"}]',
            }
        ]
    }

    assert validate_sample(sample, "popup", _SCHEMA) == []


def test_validate_schema_rejects_malformed_multi_tool_array():
    sample = _make_sample(
        '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
        '{"arguments":{"action":"打开","device":"空调"}}]'
    )

    errs = validate_sample_schema(sample, _SCHEMA)
    assert any("failed to parse tool call content" in e for e in errs)


def test_validate_schema_ignores_popup_choice_arrays():
    sample = _make_sample('[{"index":1,"name":"舒适"},{"index":2,"name":"运动"}]')

    assert validate_sample_schema(sample, _SCHEMA) == []
