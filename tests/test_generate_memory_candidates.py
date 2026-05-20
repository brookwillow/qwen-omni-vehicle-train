import json

from scripts.generate_memory_candidates import (
    DEFAULT_SERVER,
    build_dialog_messages,
    parse_model_output,
    request_candidate,
    unique_candidates,
)


def _task():
    return {
        "id": "memory_rl_0001",
        "task_type": "action_flip",
        "history": [{"role": "user", "content": "关闭大灯"}],
        "current_query": "打开吧",
    }


def test_default_server_is_openai_compatible_v1():
    assert DEFAULT_SERVER.endswith("/v1")


def test_build_dialog_messages_uses_history_plus_current_user():
    messages = build_dialog_messages(_task())

    assert messages == [
        {"role": "user", "content": "关闭大灯"},
        {"role": "user", "content": "打开吧"},
    ]


def test_parse_model_output_accepts_json_and_raw_text():
    assert parse_model_output('```json\n{"a":1}\n```') == {"a": 1}
    assert parse_model_output('前缀 {"a":2} 后缀') == {"a": 2}
    assert parse_model_output("not json") == "not json"


def test_unique_candidates_deduplicates_dicts_and_strings():
    unique = unique_candidates([
        {"a": 1, "b": 2},
        {"b": 2, "a": 1},
        "raw",
        "raw",
    ])

    assert unique == [{"a": 1, "b": 2}, "raw"]


def test_request_candidate_uses_openai_client_content():
    captured = {}

    class Message:
        content = '{"name":"LightControl","arguments":{"action":"打开","device":"大灯"}}'
        tool_calls = None

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    raw = request_candidate(
        Client(),
        model="qwen-omni-lora",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
        temperature=0.7,
        timeout=30,
    )

    assert json.loads(raw)["name"] == "LightControl"
    assert captured["model"] == "qwen-omni-lora"
    assert captured["temperature"] == 0.7
