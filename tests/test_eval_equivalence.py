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


def test_ambiguous_ventilation_accepts_window_when_gt_is_climate(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    tool_ok, args_ok = eval_mod.is_action_match(
        "车里太闷了",
        "WindowControl",
        {"action": "打开", "device": "车窗"},
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "外循环"},
    )

    assert tool_ok
    assert args_ok


def test_ambiguous_ventilation_accepts_climate_when_gt_is_window(monkeypatch):
    eval_mod = _load_eval_module(monkeypatch)

    tool_ok, args_ok = eval_mod.is_action_match(
        "车里太闷了",
        "ClimateControl",
        {"action": "打开", "device": "空调", "value": "外循环"},
        "WindowControl",
        {"action": "打开", "device": "车窗"},
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
