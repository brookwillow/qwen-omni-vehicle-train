"""Tests for last-user-anchored assistant label supervision in train_thinker_lora.py."""

from train_thinker_lora import (
    build_last_assistant_labels,
    find_assistant_contents_after_last_user,
    find_last_subsequence,
)


def test_find_last_subsequence_basic():
    assert find_last_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2


def test_find_last_subsequence_returns_last():
    assert find_last_subsequence([1, 2, 1, 2, 3], [1, 2]) == 2


def test_find_last_subsequence_not_found():
    assert find_last_subsequence([1, 2, 3], [4, 5]) == -1


def test_build_last_assistant_labels_single():
    input_ids = [10, 20, 30, 40, 50]
    target_ids = [30, 40]
    labels, matched = build_last_assistant_labels(input_ids, target_ids)
    assert matched is True
    assert labels == [-100, -100, 30, 40, -100]


def test_after_last_user_single_turn():
    """Single-turn: user -> assistant(tool_call) — supervises the tool_call."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "open window"},
        {"role": "assistant", "content": '{"name":"WindowControl","arguments":{"action":"打开"}}'},
    ]
    contents = find_assistant_contents_after_last_user(messages)
    assert len(contents) == 1
    assert "WindowControl" in contents[0]


def test_after_last_user_with_tool_result():
    """user -> tool_call -> tool -> TTS — supervises only the final assistant."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "open window"},
        {"role": "assistant", "content": '{"name":"WindowControl","arguments":{"action":"打开"}}'},
        {"role": "tool", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "已打开车窗"},
    ]
    contents = find_assistant_contents_after_last_user(messages)
    assert contents == ["已打开车窗"]


def test_after_last_user_multiturn_history_excluded():
    """Multi-turn: history assistant TTS should NOT be supervised."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "播放周杰伦的歌"},
        {"role": "assistant", "content": "好的，正在为您播放周杰伦的歌曲"},  # history TTS — skip
        {"role": "user", "content": "打开车窗"},  # last user
        {"role": "assistant", "content": '{"name":"WindowControl","arguments":{"action":"打开","device":"车窗"}}'},
    ]
    contents = find_assistant_contents_after_last_user(messages)
    assert len(contents) == 1
    assert "WindowControl" in contents[0]
    # The history TTS "好的，正在为您播放" should NOT appear
    assert not any("播放" in c for c in contents)


def test_after_last_user_no_user_message():
    """Standalone tool-result-response sample (no user) — returns empty,
    but the preprocessor falls back to last-assistant supervision."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "assistant", "content": '{"name":"WindowControl"}'},
        {"role": "tool", "content": '{"status":"success"}'},
        {"role": "assistant", "content": "已打开车窗"},
    ]
    contents = find_assistant_contents_after_last_user(messages)
    # No user message → empty (preprocessor handles the fallback)
    assert contents == []


def test_after_last_user_skips_empty():
    messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": "hello"},
    ]
    contents = find_assistant_contents_after_last_user(messages)
    assert contents == ["hello"]


def test_build_last_assistant_labels_uses_final_reject_occurrence():
    system_reject = [78413]
    assistant_reject = [78413]
    input_ids = [1, *system_reject, 2, 3, *assistant_reject, 4]

    labels, matched = build_last_assistant_labels(input_ids, assistant_reject)

    assert matched is True
    assert labels == [-100, -100, -100, -100, 78413, -100]
