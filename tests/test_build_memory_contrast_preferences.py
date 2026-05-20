import json

from scripts.build_memory_contrast_preferences import (
    build_memory_contrast_preferences,
    is_tool_json,
)


def _task(task_type="cross_tool_interrupt"):
    return {
        "id": "memory_rl_0005",
        "task_type": task_type,
        "history": [
            {"role": "user", "content": "打开主驾车窗"},
            {"role": "assistant", "content": "好的，已为您打开主驾车窗。"},
            {"role": "user", "content": "打开前雾灯"},
            {"role": "assistant", "content": "好的，前雾灯已打开。"},
        ],
        "current_query": "再关掉吧",
        "expected": {
            "relevant_memory": {
                "resolved_intent": "打开前雾灯",
                "tool_call": {"name": "LightControl", "arguments": {"action": "打开", "device": "前雾灯"}},
            },
            "ignore_memory": ["打开主驾车窗"],
            "target_tool_call": {"name": "LightControl", "arguments": {"action": "关闭", "device": "前雾灯"}},
        },
    }


def test_is_tool_json_requires_name_and_arguments():
    assert is_tool_json({"name": "WindowControl", "arguments": {}}) is True
    assert is_tool_json("Reject") is False
    assert is_tool_json({"name": "WindowControl"}) is False


def test_build_memory_contrast_preferences_prefers_expected_over_wrong_history_tool():
    rows = build_memory_contrast_preferences([_task()])

    assert len(rows) == 1
    assert rows[0]["task_type"] == "memory_contrast"
    assert rows[0]["source_task_type"] == "cross_tool_interrupt"
    assert rows[0]["chosen"] == {"name": "LightControl", "arguments": {"action": "关闭", "device": "前雾灯"}}
    assert rows[0]["rejected"] == {
        "name": "WindowControl",
        "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
    }
    json.dumps(rows[0], ensure_ascii=False)


def test_build_memory_contrast_preferences_skips_non_tool_chosen():
    task = _task("clarify_missing")
    task["expected"] = {"target_tool_call": None, "clarification": "请问您想调节哪个功能？"}

    assert build_memory_contrast_preferences([task]) == []
