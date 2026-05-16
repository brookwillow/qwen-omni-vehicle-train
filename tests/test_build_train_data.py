import json
import random
from pathlib import Path

from build_train_data import load_weighted_splits, match_weight_selector, parse_kv_args


def _write_jsonl(path: Path, queries: list[str]) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "{\"name\":\"WindowControl\",\"arguments\":{\"action\":\"打开\",\"device\":\"车窗\"}}"},
            ]
        }
        for query in queries
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_match_weight_selector_accepts_stem_path_and_glob():
    path = Path("data/splits/hard_cases/window_args.jsonl")

    assert match_weight_selector(path, "window_args")
    assert match_weight_selector(path, "hard_cases/window_args.jsonl")
    assert match_weight_selector(path, "hard_cases/*.jsonl")
    assert not match_weight_selector(path, "ClimateControl")


def test_load_weighted_splits_oversamples_matching_file(tmp_path):
    normal = tmp_path / "WindowControl.jsonl"
    hard_dir = tmp_path / "hard_cases"
    hard_dir.mkdir()
    hard = hard_dir / "window_args.jsonl"
    _write_jsonl(normal, ["普通开窗"])
    _write_jsonl(hard, ["强对比开窗"])

    samples, counts = load_weighted_splits(
        [normal, hard],
        oversample={},
        max_per_type={},
        sample_weights={"hard_cases/*.jsonl": 3.0},
        rng=random.Random(7),
    )

    queries = [sample["messages"][0]["content"] for sample in samples]
    assert queries.count("普通开窗") == 1
    assert queries.count("强对比开窗") == 3
    assert counts[str(normal)] == 1
    assert counts[str(hard)] == 3


def test_parse_kv_args_preserves_string_selectors():
    assert parse_kv_args(["hard_cases/*.jsonl:2.5"]) == {"hard_cases/*.jsonl": 2.5}
