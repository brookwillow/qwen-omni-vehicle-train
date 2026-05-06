import json
from pathlib import Path

from scripts.validate_splits import load_schema, validate_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "blackbox_priority_aug.jsonl"


def _load_samples() -> list[dict]:
    return [
        json.loads(line)
        for line in SPLIT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_blackbox_priority_augmentation_split_exists_with_enough_samples():
    samples = _load_samples()

    assert len(samples) >= 80


def test_blackbox_priority_augmentation_split_is_schema_valid():
    schema = load_schema(str(PROJECT_ROOT / "data" / "tools.json"))

    issues = []
    for index, sample in enumerate(_load_samples(), start=1):
        issues.extend(validate_sample(sample, f"{SPLIT_PATH.name}:{index}", schema))

    assert issues == []


def test_blackbox_priority_augmentation_targets_known_error_clusters():
    samples = _load_samples()
    contents = "\n".join(
        msg["content"]
        for sample in samples
        for msg in sample["messages"]
        if msg["role"] == "assistant"
    )

    for tool in ("ClimateControl", "ProfileControl", "WindowControl", "LightControl"):
        assert contents.count(f"Action: {tool}") >= 12
