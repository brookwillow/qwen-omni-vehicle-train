import json
from collections import Counter
from pathlib import Path

from scripts.generate_memory_rl_tasks import generate_tasks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_TASKS_PATH = PROJECT_ROOT / "data" / "rl" / "memory_tasks.jsonl"


def _load_memory_tasks() -> list[dict]:
    return [
        json.loads(line)
        for line in MEMORY_TASKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_memory_rl_seed_dataset_has_expected_size_and_unique_ids():
    tasks = _load_memory_tasks()

    assert len(tasks) == 500
    assert len({task["id"] for task in tasks}) == 500


def test_memory_rl_seed_dataset_covers_core_scenarios():
    tasks = _load_memory_tasks()
    counts = Counter(task["task_type"] for task in tasks)

    assert set(counts) == {
        "recent_related_inheritance",
        "action_flip",
        "position_override",
        "current_override",
        "cross_tool_interrupt",
        "noise_no_inherit",
        "clarify_missing",
    }
    assert min(counts.values()) >= 70


def test_memory_rl_seed_dataset_includes_long_histories():
    tasks = _load_memory_tasks()
    turns = [task["history_turns"] for task in tasks]

    assert sum(2 <= n <= 3 for n in turns) == 150
    assert sum(4 <= n <= 6 for n in turns) == 200
    assert sum(7 <= n <= 10 for n in turns) == 100
    assert sum(n >= 11 for n in turns) == 50


def test_memory_rl_seed_dataset_schema_shape():
    for task in _load_memory_tasks():
        assert task["id"].startswith("memory_rl_")
        assert isinstance(task["history"], list)
        assert task["history"]
        assert all(msg["role"] in {"user", "assistant"} for msg in task["history"])
        assert isinstance(task["current_query"], str) and task["current_query"]

        expected = task["expected"]
        assert expected["memory_decision"] in {
            "use_recent_related",
            "use_related_slot",
            "current_override",
            "ignore_history_noise",
            "need_clarification",
        }
        assert "resolved_intent" in expected
        assert "should_use_history" in expected
        if expected["target_tool_call"] is not None:
            assert "name" in expected["target_tool_call"]
            assert "arguments" in expected["target_tool_call"]


def test_generator_is_deterministic_for_same_seed():
    assert generate_tasks(20, seed=7) == generate_tasks(20, seed=7)
