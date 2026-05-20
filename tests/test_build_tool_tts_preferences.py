import json

from scripts.build_tool_tts_preferences import (
    build_tool_tts_preferences,
    render_tts_from_tool_call,
    target_tool_call,
)


def _task(tool_name="WindowControl"):
    tool = {"name": tool_name, "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"}}
    return {
        "id": "memory_rl_0001",
        "task_type": "action_flip",
        "history": [{"role": "user", "content": "打开主驾车窗"}],
        "current_query": "关上吧",
        "expected": {"target_tool_call": tool},
    }


def test_render_tts_from_window_tool_call():
    assert (
        render_tts_from_tool_call(
            {"name": "WindowControl", "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"}}
        )
        == "好的，已为您关闭主驾车窗。"
    )


def test_target_tool_call_skips_noise():
    task = _task("NoiseDoNotAct")
    task["expected"]["target_tool_call"] = {"name": "NoiseDoNotAct", "arguments": {}}

    assert target_tool_call(task) is None


def test_build_tool_tts_preferences_uses_tool_json_as_chosen():
    rows = build_tool_tts_preferences([_task()])

    assert len(rows) == 1
    assert rows[0]["task_type"] == "tool_vs_tts"
    assert rows[0]["chosen"] == {"name": "WindowControl", "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"}}
    assert rows[0]["rejected"] == "好的，已为您关闭主驾车窗。"
    assert rows[0]["rejected_verdict"]["tool_or_response_correct"] is False
    json.dumps(rows[0], ensure_ascii=False)
