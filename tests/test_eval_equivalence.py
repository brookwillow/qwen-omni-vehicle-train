import importlib
import sys
import types


def _load_eval_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=lambda model, _: model)),
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        types.SimpleNamespace(process_mm_info=lambda *_, **__: (None, None, None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            Qwen2_5OmniForConditionalGeneration=types.SimpleNamespace(),
            Qwen2_5OmniProcessor=types.SimpleNamespace(),
        ),
    )
    sys.modules.pop("eval", None)
    return importlib.import_module("eval")


def test_ambiguous_ventilation_rejects_window_when_gt_is_climate(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    tool_ok, args_ok = eval_mod.is_action_match(
        "车里太闷了",
        "WindowControl",
        {"action": "打开", "device": "车窗"},
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "外循环"},
    )

    assert not tool_ok
    assert not args_ok


def test_ambiguous_ventilation_accepts_climate_action_variants(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    tool_ok, args_ok = eval_mod.is_action_match(
        "车里太闷了",
        "ClimateControl",
        {"action": "调到", "device": "空调", "value": "外循环"},
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "外循环"},
    )

    assert tool_ok
    assert args_ok


def test_explicit_window_query_does_not_accept_climate_equivalence(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    tool_ok, args_ok = eval_mod.is_action_match(
        "车里有点闷，帮我打开车窗",
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "外循环"},
        "WindowControl",
        {"action": "打开", "device": "车窗"},
    )

    assert not tool_ok
    assert not args_ok


def test_eval_reports_raw_model_output_mode(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    assert eval_mod.EVALUATION_MODE == "raw_model_output"
    assert eval_mod.POSTPROCESS_APPLIED is False


def test_eval_skips_multi_intent_rows(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    assert eval_mod.should_skip_eval_row(
        {
            "intent": "多意图",
            "expected_tool_calls": [
                {"name": "WindowControl", "arguments": {}},
            ],
        }
    )
    assert eval_mod.should_skip_eval_row(
        {
            "expected_tool_calls": [
                {"name": "WindowControl", "arguments": {}},
                {"name": "LightControl", "arguments": {}},
            ],
        }
    )
    assert not eval_mod.should_skip_eval_row(
        {
            "intent": "明确指令",
            "expected_tool_calls": [
                {"name": "WindowControl", "arguments": {}},
            ],
        }
    )


def test_eval_can_include_multi_tool_rows(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    row = {
        "intent": "多意图",
        "expected_tool_calls": [
            {"name": "WindowControl", "arguments": {}},
            {"name": "LightControl", "arguments": {}},
        ],
    }

    assert eval_mod.should_skip_eval_row(row)
    assert not eval_mod.should_skip_eval_row(row, include_multi_tool=True)


def test_parse_actions_accepts_json_array(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    calls, pred_type = eval_mod.parse_actions(
        '[{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}},'
        '{"name":"LightControl","arguments":{"action":"关闭","device":"阅读灯"}}]'
    )

    assert pred_type == "Action"
    assert calls == [
        ("WindowControl", {"action": "打开", "device": "车窗"}),
        ("LightControl", {"action": "关闭", "device": "阅读灯"}),
    ]


def test_multi_tool_match_is_orderless_by_default(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    pred = [
        ("LightControl", {"action": "关闭", "device": "阅读灯"}),
        ("WindowControl", {"action": "打开", "device": "车窗"}),
    ]
    gt = [
        ("WindowControl", {"action": "打开", "device": "车窗"}),
        ("LightControl", {"action": "关闭", "device": "阅读灯"}),
    ]

    assert eval_mod.are_action_calls_match("打开车窗，关闭阅读灯", pred, gt) == (True, True)


def test_multi_tool_match_can_require_order(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    pred = [
        ("LightControl", {"action": "关闭", "device": "阅读灯"}),
        ("WindowControl", {"action": "打开", "device": "车窗"}),
    ]
    gt = [
        ("WindowControl", {"action": "打开", "device": "车窗"}),
        ("LightControl", {"action": "关闭", "device": "阅读灯"}),
    ]

    assert eval_mod.are_action_calls_match("先打开车窗，再关闭阅读灯", pred, gt, ordered=True) == (False, False)
