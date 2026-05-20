import json

from scripts.build_memory_preferences import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    append_jsonl,
    apply_schema_score_guard,
    build_verifier_messages,
    candidate_items_for_task,
    candidate_items_from_synthetic,
    choose_preference,
    choose_preference_with_reason,
    compact_json,
    extract_json_object,
    expected_assistant_output,
    find_choices,
    group_candidate_rows,
    make_synthetic_rejected,
    post_chat_completion,
    process_task,
    response_content,
    is_tool_call_candidate,
    tool_from_intent_text,
    write_jsonl,
)


def _task(task_type="action_flip"):
    return {
        "id": "memory_rl_0001",
        "task_type": task_type,
        "history": [{"role": "user", "content": "关闭大灯"}],
        "current_query": "打开吧",
        "expected": {
            "memory_decision": "use_recent_related",
            "relevant_memory": {
                "resolved_intent": "关闭大灯",
                "tool_call": {"name": "LightControl", "arguments": {"action": "关闭", "device": "大灯"}},
            },
            "ignore_memory": [],
            "resolved_intent": "打开大灯",
            "target_tool_call": {"name": "LightControl", "arguments": {"action": "打开", "device": "大灯"}},
            "should_use_history": True,
        },
    }


def test_extract_json_object_accepts_plain_and_fenced_json():
    assert extract_json_object('{"score":9,"reason":"ok"}')["score"] == 9
    assert extract_json_object('```json\n{"score":3}\n```')["score"] == 3


def test_default_verifier_endpoint_uses_dashscope_bailian():
    assert DEFAULT_BASE_URL == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert DEFAULT_MODEL == "qwen3.6-plus"


def test_compact_json_preserves_strings_and_compacts_objects():
    assert compact_json(" hello ") == "hello"
    assert compact_json({"a": "车窗", "b": 1}) == '{"a":"车窗","b":1}'


def test_build_verifier_messages_contains_task_and_candidate():
    messages = build_verifier_messages("规则", _task(), {"memory_decision": "x"})

    assert messages[0] == {"role": "system", "content": "规则"}
    assert "memory_rl_0001" in messages[1]["content"]
    assert "memory_decision" in messages[1]["content"]


def test_post_chat_completion_uses_openai_client_shape():
    captured = {}

    class Completion:
        def model_dump(self):
            return {"choices": [{"message": {"content": '{"score":9}'}}]}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Completion()

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    resp = post_chat_completion(
        Client(),
        "https://unused",
        "secret",
        "qwen3.6-plus",
        [{"role": "user", "content": "hi"}],
        temperature=0,
        max_tokens=128,
        timeout=30,
    )

    assert captured["model"] == "qwen3.6-plus"
    assert captured["messages"][0]["content"] == "hi"
    assert response_content(resp) == '{"score":9}'


def test_response_content_accepts_wrapped_choices():
    resp = {"data": {"choices": [{"message": {"content": '{"score":8}'}}]}}

    assert find_choices(resp)[0]["message"]["content"] == '{"score":8}'
    assert response_content(resp) == '{"score":8}'


def test_response_content_error_includes_preview():
    try:
        response_content({"code": "InvalidRequest", "message": "bad model"})
    except ValueError as exc:
        assert "InvalidRequest" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_schema_score_guard_caps_tts_when_tool_expected():
    verdict = apply_schema_score_guard(
        _task(),
        "好的，已为您打开大灯。",
        {"score": 10, "reason": "语义正确", "tool_or_response_correct": True, "arguments_correct": True},
    )

    assert verdict["score"] == 4
    assert verdict["tool_or_response_correct"] is False
    assert verdict["arguments_correct"] is False
    assert verdict["local_score_guard"] == "expected_tool_requires_tool_json"


def test_schema_score_guard_allows_tool_json_when_tool_expected():
    candidate = {"name": "LightControl", "arguments": {"action": "打开", "device": "大灯"}}
    verdict = apply_schema_score_guard(_task(), candidate, {"score": 10, "reason": "ok"})

    assert verdict["score"] == 10
    assert "local_score_guard" not in verdict
    assert is_tool_call_candidate(candidate) is True


def test_make_synthetic_rejected_for_action_flip_uses_wrong_history_action():
    rejected = make_synthetic_rejected(_task("action_flip"))

    assert rejected == {"name": "LightControl", "arguments": {"action": "关闭", "device": "大灯"}}


def test_make_synthetic_rejected_for_noise_inherits_history():
    task = _task("noise_no_inherit")
    rejected = make_synthetic_rejected(task)

    assert rejected == {"name": "LightControl", "arguments": {"action": "关闭", "device": "大灯"}}


def test_expected_assistant_output_uses_final_output_shape():
    assert expected_assistant_output(_task()) == {
        "name": "LightControl",
        "arguments": {"action": "打开", "device": "大灯"},
    }

    clarify_task = _task("clarify_missing")
    clarify_task["expected"] = {
        "target_tool_call": None,
        "clarification": "请问您想调节哪个功能？",
    }
    assert expected_assistant_output(clarify_task) == "请问您想调节哪个功能？"


def test_tool_from_intent_text_parses_generated_ignore_memory():
    assert tool_from_intent_text("打开主驾车窗") == {
        "name": "WindowControl",
        "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
    }
    assert tool_from_intent_text("调低导航音量") == {
        "name": "VoiceControl",
        "arguments": {"action": "调低", "feature": "导航音量"},
    }


def test_cross_tool_synthetic_negative_uses_ignored_memory():
    task = _task("cross_tool_interrupt")
    task["expected"]["ignore_memory"] = ["打开主驾车窗"]

    rejected = make_synthetic_rejected(task)

    assert rejected == {
        "name": "WindowControl",
        "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
    }


def test_choose_preference_keeps_high_confidence_pair():
    task = _task()
    chosen = {"memory_decision": "use_recent_related"}
    rejected = {"memory_decision": "current_override"}

    pref = choose_preference(
        task,
        [
            {"source": "a", "candidate": rejected, "verdict": {"score": 2, "reason": "bad"}},
            {"source": "b", "candidate": chosen, "verdict": {"score": 9, "reason": "good"}},
        ],
        min_chosen_score=8,
        max_rejected_score=4,
        min_gap=4,
    )

    assert pref is not None
    assert pref["chosen"] == chosen
    assert pref["rejected"] == rejected
    assert pref["chosen_score"] == 9
    assert pref["rejected_score"] == 2


def test_choose_preference_drops_small_gap():
    assert choose_preference(
        _task(),
        [
            {"candidate": {"a": 1}, "verdict": {"score": 8}},
            {"candidate": {"a": 2}, "verdict": {"score": 6}},
        ],
        min_chosen_score=8,
        max_rejected_score=6,
        min_gap=4,
    ) is None


def test_choose_preference_with_reason_reports_skip_reason():
    decision = choose_preference_with_reason(
        _task(),
        [
            {"source": "best", "candidate": {"a": 1}, "verdict": {"score": 8}},
            {"source": "worst", "candidate": {"a": 2}, "verdict": {"score": 6}},
        ],
        min_chosen_score=8,
        max_rejected_score=6,
        min_gap=4,
    )

    assert decision["preference"] is None
    assert decision["skip_reason"] == "score_gap_below_threshold"
    assert decision["best_score"] == 8
    assert decision["worst_score"] == 6
    assert decision["score_gap"] == 2


def test_group_candidate_rows_supports_candidate_list_and_single_candidate():
    rows = [
        {"id": "x", "candidates": [{"a": 1}, {"a": 2}]},
        {"task_id": "x", "source": "sampled", "candidate": {"a": 3}},
    ]

    grouped = group_candidate_rows(rows)

    assert len(grouped["x"]["candidates"]) == 3
    assert grouped["x"]["candidates"][2]["source"] == "sampled"


def test_candidate_file_mode_mixes_synthetic_negative_by_default():
    task = _task("action_flip")
    groups = group_candidate_rows([
        {"id": "memory_rl_0001", "candidate": {"name": "LightControl", "arguments": {"action": "打开"}}}
    ])

    items = candidate_items_for_task(task, groups)

    assert [item["source"] for item in items] == ["candidate", "synthetic_hard_negative"]


def test_synthetic_mode_uses_final_assistant_outputs():
    items = candidate_items_from_synthetic(_task("action_flip"))

    assert items[0]["candidate"] == {"name": "LightControl", "arguments": {"action": "打开", "device": "大灯"}}
    assert items[1]["candidate"] == {"name": "LightControl", "arguments": {"action": "关闭", "device": "大灯"}}


def test_candidate_file_mode_can_disable_synthetic_negative():
    task = _task("action_flip")
    groups = group_candidate_rows([
        {"id": "memory_rl_0001", "candidate": {"name": "LightControl", "arguments": {"action": "打开"}}}
    ])

    items = candidate_items_for_task(task, groups, include_synthetic_negative=False)

    assert [item["source"] for item in items] == ["candidate"]


def test_write_jsonl_overwrites_and_append_jsonl_appends(tmp_path):
    path = tmp_path / "out.jsonl"

    write_jsonl([{"a": 1}], path)
    write_jsonl([{"a": 2}], path)
    append_jsonl([{"a": 3}], path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"a": 2}, {"a": 3}]


def test_preference_rows_are_json_serializable():
    pref = choose_preference(
        _task(),
        [
            {"candidate": {"a": 1}, "verdict": {"score": 9}},
            {"candidate": {"a": 2}, "verdict": {"score": 1}},
        ],
        min_chosen_score=8,
        max_rejected_score=4,
        min_gap=4,
    )

    json.dumps(pref, ensure_ascii=False)


def test_process_task_skips_missing_candidates():
    idx, pref, audit, msg = process_task(
        1,
        1,
        _task(),
        [],
        "prompt",
        object(),
        "key",
        object(),
    )

    assert idx == 1
    assert pref is None
    assert audit["skip_reason"] == "no_candidates"
    assert "[skip]" in msg


def test_process_task_audit_reports_threshold_reason():
    class Args:
        min_chosen_score = 8
        max_rejected_score = 4
        min_gap = 4
        model = "unused"
        base_url = "unused"
        timeout = 1
        verifier_temperature = 0
        verifier_max_tokens = 1
        retries = 1

    class Client:
        pass

    verdicts = iter([
        {"score": 8, "reason": "ok"},
        {"score": 6, "reason": "near miss"},
    ])

    def fake_score_candidate(*_args, **_kwargs):
        return next(verdicts)

    import scripts.build_memory_preferences as bmp

    original = bmp.score_candidate
    bmp.score_candidate = fake_score_candidate
    try:
        _, pref, audit, _ = process_task(
            1,
            1,
            _task(),
            [{"source": "a", "candidate": {"a": 1}}, {"source": "b", "candidate": {"a": 2}}],
            "prompt",
            Args(),
            "key",
            Client(),
        )
    finally:
        bmp.score_candidate = original

    assert pref is None
    assert audit["skip_reason"] == "rejected_score_above_threshold"
    assert audit["best_score"] == 8
    assert audit["worst_score"] == 6
