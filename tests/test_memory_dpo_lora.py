import json

import pytest

from train_memory_dpo_lora import (
    expand_preference_files,
    format_memory_messages,
    format_memory_prompt,
    load_peft_adapter,
    load_preference_rows,
    load_preference_rows_many,
    normalize_response,
    normalize_token_ids,
    split_train_eval,
)


def test_normalize_response_compacts_json_objects():
    assert normalize_response({"b": 1, "a": "车窗"}) == '{"b":1,"a":"车窗"}'


def test_format_memory_prompt_uses_history_and_current_query():
    row = {
        "history": [
            {"role": "user", "content": "打开主驾车窗"},
            {"role": "assistant", "content": "好的，已打开主驾车窗。"},
        ],
        "current_query": "关上吧",
        "task_type": "action_flip",
    }

    prompt = format_memory_prompt(row)

    assert "当前轮最终 assistant 输出" in prompt
    assert "打开主驾车窗" in prompt
    assert "关上吧" in prompt
    assert prompt.endswith("assistant:")


def test_format_memory_messages_matches_chat_history_shape():
    row = load_preference_rows_from_dict(
        {
            "history": [
                {"role": "user", "content": "打开主驾车窗"},
                {"role": "assistant", "content": "好的，已打开主驾车窗。"},
            ],
            "current_query": "关上吧",
            "chosen": {"name": "WindowControl", "arguments": {"action": "关闭", "device": "车窗", "position": "主驾"}},
            "rejected": {"name": "WindowControl", "arguments": {"action": "打开", "device": "车窗", "position": "主驾"}},
        }
    )

    messages = format_memory_messages(row, "系统提示")

    assert messages == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "打开主驾车窗"},
        {"role": "assistant", "content": "好的，已打开主驾车窗。"},
        {"role": "user", "content": "关上吧"},
    ]


def test_normalize_token_ids_accepts_common_chat_template_shapes():
    class TensorLike:
        def tolist(self):
            return [[1, 2, 3]]

    class BatchEncodingLike:
        def __init__(self):
            self.data = {"input_ids": [[1, 2, 3]]}

    assert normalize_token_ids([1, 2, 3]) == [1, 2, 3]
    assert normalize_token_ids({"input_ids": [[1, 2, 3]]}) == [1, 2, 3]
    assert normalize_token_ids(TensorLike()) == [1, 2, 3]
    assert normalize_token_ids(BatchEncodingLike()) == [1, 2, 3]


def test_normalize_token_ids_rejects_string_keys_shape():
    with pytest.raises(ValueError, match="list\\[int\\]"):
        normalize_token_ids(["input_ids"])


def load_preference_rows_from_dict(raw):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prefs.jsonl"
        path.write_text(json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8")
        return load_preference_rows(path)[0]


def test_load_preference_rows_supports_explicit_prompt(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "p",
                "chosen": {"memory_decision": "current_override"},
                "rejected": {"memory_decision": "use_recent_related"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_preference_rows(path)

    assert len(rows) == 1
    assert rows[0].prompt == "p"
    assert rows[0].chosen == '{"memory_decision":"current_override"}'
    assert rows[0].rejected == '{"memory_decision":"use_recent_related"}'
    assert rows[0].history == []
    assert rows[0].current_query == ""


def test_load_preference_rows_supports_memory_task_shape(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text(
        json.dumps(
            {
                "history": [{"role": "user", "content": "关闭大灯"}],
                "current_query": "打开吧",
                "chosen": '{"memory_decision":"use_recent_related"}',
                "rejected": '{"memory_decision":"current_override"}',
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_preference_rows(path)

    assert "关闭大灯" in rows[0].prompt
    assert "打开吧" in rows[0].prompt
    assert rows[0].chosen == '{"memory_decision":"use_recent_related"}'
    assert rows[0].history == [{"role": "user", "content": "关闭大灯"}]
    assert rows[0].current_query == "打开吧"


def test_load_preference_rows_rejects_identical_pair(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text('{"prompt":"p","chosen":"same","rejected":"same"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identical"):
        load_preference_rows(path)


def test_load_preference_rows_many_supports_comma_separated_files(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"prompt":"p1","chosen":"a","rejected":"b"}\n', encoding="utf-8")
    second.write_text('{"prompt":"p2","chosen":"c","rejected":"d"}\n', encoding="utf-8")

    rows = load_preference_rows_many(f"{first},{second}")

    assert [row.prompt for row in rows] == ["p1", "p2"]


def test_expand_preference_files_supports_globs(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"prompt":"p","chosen":"a","rejected":"b"}\n', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"prompt":"p","chosen":"a","rejected":"b"}\n', encoding="utf-8")

    paths = expand_preference_files(str(tmp_path / "*.jsonl"))

    assert [path.name for path in paths] == ["a.jsonl", "b.jsonl"]


def test_split_train_eval_is_deterministic():
    rows = [
        load_preference_rows_from_dict({"prompt": str(i), "chosen": str(i), "rejected": str(i + 1)})
        for i in range(10)
    ]

    first = split_train_eval(rows, val_ratio=0.2, seed=3)
    second = split_train_eval(rows, val_ratio=0.2, seed=3)

    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2


def test_load_peft_adapter_falls_back_when_is_trainable_is_unsupported():
    class Param:
        def __init__(self):
            self.requires_grad = False

    class FakeModel:
        def __init__(self):
            self.lora = Param()
            self.base = Param()

        def named_parameters(self):
            return [
                ("base.weight", self.base),
                ("adapter.lora_A.weight", self.lora),
            ]

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base_model, adapter_dir, **kwargs):
            if "is_trainable" in kwargs:
                raise TypeError("unsupported")
            return base_model

    model = FakeModel()

    loaded = load_peft_adapter(FakePeftModel, model, "adapter", trainable=True)

    assert loaded is model
    assert model.lora.requires_grad is True
    assert model.base.requires_grad is False
