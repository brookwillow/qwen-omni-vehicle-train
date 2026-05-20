#!/usr/bin/env python3
"""Generate memory-compression/use tasks for LLM-verifier RL experiments.

The output is not part of the supervised tool-calling train split.  It is a
source dataset for candidate generation and LLM-as-verifier preference building.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("data/rl/memory_tasks.jsonl")


@dataclass(frozen=True)
class Operation:
    user: str
    assistant: str
    tool_call: dict[str, Any]
    resolved_intent: str


OPS: list[Operation] = [
    Operation(
        "打开主驾车窗",
        "好的，已为您打开主驾车窗。",
        {"name": "WindowControl", "arguments": {"action": "打开", "device": "车窗", "position": "主驾"}},
        "打开主驾车窗",
    ),
    Operation(
        "关闭副驾车窗",
        "好的，已为您关闭副驾车窗。",
        {"name": "WindowControl", "arguments": {"action": "关闭", "device": "车窗", "position": "副驾"}},
        "关闭副驾车窗",
    ),
    Operation(
        "把后排左侧车窗打开",
        "好的，已打开后排左侧车窗。",
        {"name": "WindowControl", "arguments": {"action": "打开", "device": "车窗", "position": "第二排左侧"}},
        "打开后排左侧车窗",
    ),
    Operation(
        "关闭大灯",
        "好的，已为您关闭大灯。",
        {"name": "LightControl", "arguments": {"action": "关闭", "device": "大灯"}},
        "关闭大灯",
    ),
    Operation(
        "打开前雾灯",
        "好的，前雾灯已打开。",
        {"name": "LightControl", "arguments": {"action": "打开", "device": "前雾灯"}},
        "打开前雾灯",
    ),
    Operation(
        "把空调调到制冷",
        "好的，空调已调到制冷。",
        {"name": "ClimateControl", "arguments": {"action": "调到", "device": "空调", "value": "制冷"}},
        "空调调到制冷",
    ),
    Operation(
        "空调温度调高一点",
        "好的，已调高空调温度。",
        {"name": "ClimateControl", "arguments": {"action": "调高", "device": "空调", "feature": "温度"}},
        "调高空调温度",
    ),
    Operation(
        "把副驾座椅加热打开",
        "好的，副驾座椅加热已打开。",
        {"name": "SeatControl", "arguments": {"action": "打开", "device": "座椅", "feature": "制热", "position": "副驾"}},
        "打开副驾座椅加热",
    ),
    Operation(
        "主驾座椅往后一点",
        "好的，已将主驾座椅往后调。",
        {"name": "SeatControl", "arguments": {"action": "调后", "device": "座椅", "position": "主驾"}},
        "主驾座椅调后",
    ),
    Operation(
        "关闭左侧侧滑门",
        "好的，左侧侧滑门已关闭。",
        {"name": "GateControl", "arguments": {"action": "关闭", "device": "侧滑门", "position": "左侧"}},
        "关闭左侧侧滑门",
    ),
    Operation(
        "打开后备箱",
        "好的，后备箱已打开。",
        {"name": "GateControl", "arguments": {"action": "打开", "device": "后备箱"}},
        "打开后备箱",
    ),
    Operation(
        "把导航音量调低一点",
        "好的，已调低导航音量。",
        {"name": "VoiceControl", "arguments": {"action": "调低", "feature": "导航音量"}},
        "调低导航音量",
    ),
]

CHATTER = [
    ("今天天气怎么样", "今天的天气适合出行。"),
    ("讲个笑话", "好的，给您讲一个轻松的笑话。"),
    ("帮我看看新闻", "好的，我来为您播报新闻。"),
    ("刚才那首歌叫什么", "我帮您看一下当前播放信息。"),
    ("谢谢", "不客气。"),
    ("算了", "好的。"),
]

NOISE_QUERIES = ["嗯", "对", "不是那个", "先这样", "等一下", "那个"]


def _pair(op: Operation) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": op.user},
        {"role": "assistant", "content": op.assistant},
    ]


def _chat_pair(rng: random.Random) -> list[dict[str, str]]:
    user, assistant = rng.choice(CHATTER)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def _target_from_op(op: Operation, decision: str, resolved_intent: str | None = None) -> dict[str, Any]:
    return {
        "memory_decision": decision,
        "relevant_memory": {
            "resolved_intent": op.resolved_intent,
            "tool_call": op.tool_call,
        },
        "ignore_memory": [],
        "resolved_intent": resolved_intent or op.resolved_intent,
        "target_tool_call": op.tool_call,
        "should_use_history": decision in {"use_recent_related", "use_related_slot"},
    }


def _window_variant(base: Operation, target_position: str, current_query: str) -> dict[str, Any]:
    args = dict(base.tool_call["arguments"])
    args["position"] = target_position
    resolved = f"{args['action']}{target_position}车窗"
    return {
        "memory_decision": "use_related_slot",
        "relevant_memory": {
            "resolved_intent": base.resolved_intent,
            "tool_call": base.tool_call,
        },
        "ignore_memory": [],
        "resolved_intent": resolved,
        "target_tool_call": {"name": "WindowControl", "arguments": args},
        "should_use_history": True,
        "slot_override": {"position": target_position},
    }


def _add_distractors(history: list[dict[str, str]], rng: random.Random, pairs: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for _ in range(pairs):
        if rng.random() < 0.55:
            out.extend(_chat_pair(rng))
        else:
            out.extend(_pair(rng.choice(OPS)))
    out.extend(history)
    return out


def _history_pair_count(history: list[dict[str, str]]) -> int:
    return sum(1 for msg in history if msg.get("role") == "user")


def _with_length_bucket(
    history: list[dict[str, str]],
    rng: random.Random,
    bucket: tuple[int, int],
) -> list[dict[str, str]]:
    current_pairs = _history_pair_count(history)
    target_pairs = rng.randint(*bucket)
    if current_pairs >= target_pairs:
        return history
    return _add_distractors(history, rng, target_pairs - current_pairs)


def _make_recent_related(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    distractor = rng.choice(OPS)
    target = rng.choice([op for op in OPS if op.tool_call != distractor.tool_call])
    history = _pair(distractor) + _pair(target)
    action = target.tool_call["arguments"].get("action", "")
    query = "再来一次" if action in {"打开", "关闭"} else "再调一下"
    return _with_length_bucket(history, rng, bucket), query, _target_from_op(target, "use_recent_related")


def _make_action_flip(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    base = rng.choice([op for op in OPS if op.tool_call["arguments"].get("action") in {"打开", "关闭"}])
    args = dict(base.tool_call["arguments"])
    args["action"] = "关闭" if args["action"] == "打开" else "打开"
    query = "关掉吧" if args["action"] == "关闭" else "打开吧"
    resolved = base.resolved_intent.replace("打开", args["action"]).replace("关闭", args["action"])
    target = {
        "name": base.tool_call["name"],
        "arguments": args,
    }
    expected = {
        "memory_decision": "use_recent_related",
        "relevant_memory": {"resolved_intent": base.resolved_intent, "tool_call": base.tool_call},
        "ignore_memory": [],
        "resolved_intent": resolved,
        "target_tool_call": target,
        "should_use_history": True,
        "slot_override": {"action": args["action"]},
    }
    return _with_length_bucket(_pair(base), rng, bucket), query, expected


def _make_position_override(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    base = OPS[0] if rng.random() < 0.5 else OPS[2]
    target_position, query = rng.choice([
        ("副驾", "副驾的也一样"),
        ("第二排右侧", "后排右边也这样"),
        ("主驾", "主驾也照这个来"),
    ])
    expected = _window_variant(base, target_position, query)
    return _with_length_bucket(_pair(base), rng, bucket), query, expected


def _make_current_override(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    old = rng.choice(OPS)
    current = rng.choice([op for op in OPS if op.tool_call["name"] != old.tool_call["name"]])
    expected = _target_from_op(current, "current_override")
    expected["relevant_memory"] = None
    expected["ignore_memory"] = [old.resolved_intent]
    expected["should_use_history"] = False
    return _with_length_bucket(_pair(old), rng, bucket), current.user, expected


def _make_cross_tool_interrupt(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    early = OPS[0]
    recent = rng.choice([op for op in OPS if op.tool_call["name"] == "LightControl"])
    history = _pair(early) + _chat_pair(rng) + _pair(recent)
    query = "再打开吧" if recent.tool_call["arguments"].get("action") == "关闭" else "再关掉吧"
    args = dict(recent.tool_call["arguments"])
    args["action"] = "打开" if query == "再打开吧" else "关闭"
    expected = {
        "memory_decision": "use_recent_related",
        "relevant_memory": {"resolved_intent": recent.resolved_intent, "tool_call": recent.tool_call},
        "ignore_memory": [early.resolved_intent],
        "resolved_intent": f"{args['action']}{args['device']}",
        "target_tool_call": {"name": recent.tool_call["name"], "arguments": args},
        "should_use_history": True,
    }
    return _with_length_bucket(history, rng, bucket), query, expected


def _make_noise_no_inherit(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    op = rng.choice(OPS)
    query = rng.choice(NOISE_QUERIES)
    expected = {
        "memory_decision": "ignore_history_noise",
        "relevant_memory": None,
        "ignore_memory": [op.resolved_intent],
        "resolved_intent": "NoiseDoNotAct",
        "target_tool_call": {"name": "NoiseDoNotAct", "arguments": {}},
        "should_use_history": False,
    }
    return _with_length_bucket(_pair(op), rng, bucket), query, expected


def _make_clarify_missing(rng: random.Random, bucket: tuple[int, int]) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    op = rng.choice(OPS)
    query = rng.choice(["调到标准模式", "切换一下模式", "调成最高"])
    expected = {
        "memory_decision": "need_clarification",
        "relevant_memory": None,
        "ignore_memory": [op.resolved_intent],
        "resolved_intent": "需要澄清目标设备或功能",
        "target_tool_call": None,
        "should_use_history": False,
        "clarification": "请问您想调节哪个功能？",
    }
    return _with_length_bucket(_pair(op), rng, bucket), query, expected


SCENARIOS = [
    ("recent_related_inheritance", _make_recent_related),
    ("action_flip", _make_action_flip),
    ("position_override", _make_position_override),
    ("current_override", _make_current_override),
    ("cross_tool_interrupt", _make_cross_tool_interrupt),
    ("noise_no_inherit", _make_noise_no_inherit),
    ("clarify_missing", _make_clarify_missing),
]

LENGTH_BUCKETS: list[tuple[tuple[int, int], int]] = [
    ((2, 3), 150),
    ((4, 6), 200),
    ((7, 10), 100),
    ((11, 12), 50),
]


def generate_tasks(n: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    buckets: list[tuple[int, int]] = []
    while len(buckets) < n:
        for bucket, count in LENGTH_BUCKETS:
            buckets.extend([bucket] * count)
    buckets = buckets[:n]
    rng.shuffle(buckets)

    for idx, bucket in enumerate(buckets, start=1):
        scenario, maker = SCENARIOS[(idx - 1) % len(SCENARIOS)]
        history, query, expected = maker(rng, bucket)
        tasks.append({
            "id": f"memory_rl_{idx:04d}",
            "task_type": scenario,
            "history_turns": _history_pair_count(history),
            "history": history,
            "current_query": query,
            "expected": expected,
            "verifier_focus": [
                "current_turn_first",
                "recent_related_memory",
                "slot_consistency",
                "ignore_distractors",
            ],
        })
    return tasks


def write_jsonl(tasks: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate memory RL task seed data.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = generate_tasks(args.count, args.seed)
    write_jsonl(tasks, Path(args.output))
    print(f"[out] {len(tasks)} memory RL tasks -> {args.output}")


if __name__ == "__main__":
    main()
