import json

import pytest

from train_memory_dpo_lora import (
    format_memory_prompt,
    load_peft_adapter,
    load_preference_rows,
    normalize_response,
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


def test_load_preference_rows_rejects_identical_pair(tmp_path):
    path = tmp_path / "prefs.jsonl"
    path.write_text('{"prompt":"p","chosen":"same","rejected":"same"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identical"):
        load_preference_rows(path)


def test_split_train_eval_is_deterministic():
    rows = [(str(i), str(i), str(i + 1)) for i in range(10)]

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
