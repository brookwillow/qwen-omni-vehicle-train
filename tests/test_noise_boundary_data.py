import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _query(row: dict) -> str:
    return row["messages"][0]["content"]


def _assistant_payload(row: dict):
    return json.loads(row["messages"][-1]["content"])


def test_noise_split_keeps_command_like_text_explicitly_non_addressed():
    noise_queries = {_query(row) for row in _rows(Path("data/splits/by_tool/NoiseDoNotAct.jsonl"))}

    ambiguous_noise = {
        "今天很热啊",
        "啊这首歌我喜欢，就这首",
        "对对对就是这个意思，你继续说",
        "继续",
        "停一下",
    }

    assert noise_queries.isdisjoint(ambiguous_noise)
    assert "乘客随口说今天很热啊，不是对小P说的" in noise_queries
    assert "乘客跟同伴说这首歌我喜欢，不是对小P下指令" in noise_queries


def test_latest_over_noise_hard_cases_are_valid_tool_requests():
    rows = _rows(Path("data/splits/hard_cases/OverNoiseRemaining_20260607.jsonl"))

    assert len(rows) == 42
    queries = {_query(row) for row in rows}
    assert "打开大灯" in queries
    assert "搜索王经理的电话" in queries
    assert "提升一下动力响应" in queries

    for row in rows:
        payload = _assistant_payload(row)
        calls = payload if isinstance(payload, list) else [payload]
        for call in calls:
            assert call["name"] not in {"NoiseDoNotAct", "Reject"}
            assert isinstance(call["arguments"], dict)
