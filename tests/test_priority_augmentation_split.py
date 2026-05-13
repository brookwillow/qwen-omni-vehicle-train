import json
from pathlib import Path

from scripts.validate_splits import load_schema, validate_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BY_TOOL_DIR = PROJECT_ROOT / "data" / "splits" / "by_tool"


def _load_samples() -> list[dict]:
    samples = []
    for path in sorted(BY_TOOL_DIR.glob("*.jsonl")):
        samples.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return samples


def test_by_tool_split_exists_with_enough_samples():
    samples = _load_samples()

    assert len(samples) >= 6000


def test_by_tool_split_is_schema_valid():
    schema = load_schema(str(PROJECT_ROOT / "data" / "tools.json"))

    issues = []
    for index, sample in enumerate(_load_samples(), start=1):
        issues.extend(validate_sample(sample, f"by_tool:{index}", schema))

    assert issues == []


def test_by_tool_split_targets_known_error_clusters():
    samples = _load_samples()
    contents = "\n".join(
        msg["content"]
        for sample in samples
        for msg in sample["messages"]
        if msg["role"] == "assistant"
    )

    for tool in ("ClimateControl", "ProfileControl", "WindowControl", "LightControl"):
        assert contents.count(f'"name":"{tool}"') >= 12


def test_phone_split_includes_builtin_contacts():
    phone_path = BY_TOOL_DIR / "PhoneControl.jsonl"
    contacts = []
    for line in phone_path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        for message in sample["messages"]:
            if message["role"] != "assistant":
                continue
            try:
                tool_call = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if tool_call.get("name") == "PhoneControl":
                contacts.append(tool_call.get("arguments", {}).get("contact"))

    for contact in ("小鹏客服", "小鹏救援", "儿童手表"):
        assert contacts.count(contact) >= 5
