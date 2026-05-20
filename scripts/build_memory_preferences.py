#!/usr/bin/env python3
"""Build memory DPO preference pairs with an LLM-as-verifier.

The script supports two sources:
- synthetic: use each task's expected output as candidate A and construct a hard
  negative candidate B from the task type.
- candidates: read pre-generated model candidates and let the verifier choose
  high/low scoring outputs.

API keys are read from environment variables and are never written to output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = os.environ.get("QWEN_PLUS_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_API_KEY_ENV = "QWEN_PLUS_API_KEY"
DEFAULT_APPID_ENV = "XPENG_AI_HUB_APPID"
DEFAULT_MODEL = os.environ.get("QWEN_PLUS_MODEL", "qwen3.6-plus")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_json(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_json_object(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(s[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError(f"verifier did not return a JSON object: {text[:200]!r}")


def build_verifier_messages(verifier_prompt: str, task: dict[str, Any], candidate: Any) -> list[dict[str, str]]:
    payload = {
        "task_id": task.get("id", ""),
        "task_type": task.get("task_type", ""),
        "history": task.get("history", []),
        "current_query": task.get("current_query", ""),
        "expected": task.get("expected"),
        "candidate": candidate,
    }
    return [
        {"role": "system", "content": verifier_prompt},
        {
            "role": "user",
            "content": (
                "请根据评分原则评价 candidate。只输出一行 JSON。\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]


def create_openai_client(api_key: str, base_url: str, appid: str = "", timeout: float = 60.0) -> Any:
    from openai import OpenAI

    headers = {}
    if appid:
        # AI Hub compatible endpoints may use one of these app id header names.
        headers["appid"] = appid
        headers["X-Appid"] = appid
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
    }
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def post_chat_completion(
    client: Any,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    _ = (base_url, api_key, timeout)  # Kept for call-site compatibility and audit readability.
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if hasattr(completion, "model_dump"):
        return completion.model_dump()
    if hasattr(completion, "dict"):
        return completion.dict()
    return completion


def response_content(resp: dict[str, Any]) -> str:
    choices = find_choices(resp)
    if not choices:
        raise ValueError(f"verifier response has no choices: {json.dumps(resp, ensure_ascii=False)[:1000]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if not content and isinstance(choices[0].get("delta"), dict):
        content = choices[0]["delta"].get("content", "")
    if not content and isinstance(message.get("reasoning_content"), str):
        content = message.get("reasoning_content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"verifier response has empty content: {json.dumps(resp, ensure_ascii=False)[:1000]}")
    return content


def find_choices(value: Any) -> list[dict[str, Any]]:
    """Find OpenAI-style choices even when providers wrap them in data/output."""
    if isinstance(value, dict):
        choices = value.get("choices")
        if isinstance(choices, list):
            return [item for item in choices if isinstance(item, dict)]
        for key in ("data", "output", "result", "response"):
            found = find_choices(value.get(key))
            if found:
                return found
    return []


def score_candidate(
    verifier_prompt: str,
    task: dict[str, Any],
    candidate: Any,
    args: argparse.Namespace,
    api_key: str,
    client: Any,
) -> dict[str, Any]:
    messages = build_verifier_messages(verifier_prompt, task, candidate)
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            resp = post_chat_completion(
                client,
                args.base_url,
                api_key,
                args.model,
                messages,
                args.temperature,
                args.max_tokens,
                args.timeout,
            )
            verdict = extract_json_object(response_content(resp))
            score = verdict.get("score")
            if not isinstance(score, (int, float)):
                raise ValueError(f"verifier score is not numeric: {verdict}")
            verdict["score"] = float(score)
            return verdict
        except Exception as exc:  # noqa: BLE001 - retry wrapper should preserve final error
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep * (attempt + 1))
    assert last_error is not None
    raise last_error


def _target_tool(task: dict[str, Any]) -> dict[str, Any] | None:
    expected = task.get("expected") or {}
    tool = expected.get("target_tool_call")
    return tool if isinstance(tool, dict) else None


def copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def expected_assistant_output(task: dict[str, Any]) -> Any:
    expected = task.get("expected") or {}
    target_tool = _target_tool(task)
    if target_tool:
        return copy_jsonable(target_tool)
    clarification = expected.get("clarification")
    if isinstance(clarification, str) and clarification.strip():
        return clarification.strip()
    resolved = expected.get("resolved_intent")
    if resolved == "Reject":
        return "Reject"
    if resolved == "NoiseDoNotAct":
        return {"name": "NoiseDoNotAct", "arguments": {}}
    return resolved or task.get("current_query", "")


def flip_action(action: Any) -> Any:
    pairs = {
        "打开": "关闭",
        "开启": "关闭",
        "关闭": "打开",
        "关": "打开",
        "开": "关闭",
        "调高": "调低",
        "调低": "调高",
        "升高": "降低",
        "降低": "升高",
        "调前": "调后",
        "调后": "调前",
        "再开": "关闭",
    }
    return pairs.get(action, action)


def mutate_tool_call(tool: dict[str, Any]) -> dict[str, Any]:
    wrong_tool = copy_jsonable(tool)
    args = wrong_tool.setdefault("arguments", {})
    if isinstance(args, dict) and args.get("action") is not None:
        flipped = flip_action(args.get("action"))
        if flipped != args.get("action"):
            args["action"] = flipped
            return wrong_tool
    name = wrong_tool.get("name")
    if name == "WindowControl":
        args["position"] = "副驾" if args.get("position") != "副驾" else "主驾"
    elif name == "LightControl":
        args["device"] = "大灯" if args.get("device") != "大灯" else "前雾灯"
    elif name == "GateControl":
        args["device"] = "后备箱" if args.get("device") != "后备箱" else "侧滑门"
    elif name == "SeatControl":
        args["position"] = "副驾" if args.get("position") != "副驾" else "主驾"
    elif name == "ClimateControl":
        args["value"] = "制热" if args.get("value") != "制热" else "制冷"
    else:
        wrong_tool = {"name": "NoiseDoNotAct", "arguments": {}}
    return wrong_tool


def tool_from_intent_text(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None

    if "车窗" in s:
        args: dict[str, Any] = {"action": "打开" if "打开" in s else "关闭", "device": "车窗"}
        if "主驾" in s:
            args["position"] = "主驾"
        elif "副驾" in s:
            args["position"] = "副驾"
        elif "后排左" in s or "第二排左" in s:
            args["position"] = "第二排左侧"
        return {"name": "WindowControl", "arguments": args}

    if "前雾灯" in s or "大灯" in s:
        device = "前雾灯" if "前雾灯" in s else "大灯"
        action = "关闭" if "关闭" in s or "关" in s else "打开"
        return {"name": "LightControl", "arguments": {"action": action, "device": device}}

    if "后备箱" in s:
        action = "关闭" if "关闭" in s or "关" in s else "打开"
        return {"name": "GateControl", "arguments": {"action": action, "device": "后备箱"}}

    if "侧滑门" in s:
        args = {"action": "关闭" if "关闭" in s or "关" in s else "打开", "device": "侧滑门"}
        if "左" in s:
            args["position"] = "左侧"
        elif "右" in s:
            args["position"] = "右侧"
        return {"name": "GateControl", "arguments": args}

    if "座椅" in s:
        args = {"device": "座椅"}
        if "副驾" in s:
            args["position"] = "副驾"
        elif "主驾" in s:
            args["position"] = "主驾"
        if "加热" in s:
            args.update({"action": "关闭" if "关闭" in s or "关" in s else "打开", "feature": "制热"})
        elif "调后" in s or "往后" in s:
            args["action"] = "调后"
        elif "调前" in s or "往前" in s:
            args["action"] = "调前"
        else:
            args["action"] = "打开" if "打开" in s else "调后"
        return {"name": "SeatControl", "arguments": args}

    if "空调" in s:
        if "调高" in s:
            return {"name": "ClimateControl", "arguments": {"action": "调高", "device": "空调", "feature": "温度"}}
        if "调低" in s:
            return {"name": "ClimateControl", "arguments": {"action": "调低", "device": "空调", "feature": "温度"}}
        if "制冷" in s:
            return {"name": "ClimateControl", "arguments": {"action": "调到", "device": "空调", "value": "制冷"}}
        if "制热" in s:
            return {"name": "ClimateControl", "arguments": {"action": "调到", "device": "空调", "value": "制热"}}

    if "导航音量" in s:
        action = "调低" if "调低" in s else "调高"
        return {"name": "VoiceControl", "arguments": {"action": action, "feature": "导航音量"}}

    return None


def first_ignored_tool(expected: dict[str, Any]) -> dict[str, Any] | None:
    for item in expected.get("ignore_memory") or []:
        tool = tool_from_intent_text(item)
        if tool:
            return tool
    return None


def make_synthetic_rejected(task: dict[str, Any]) -> Any:
    expected = task.get("expected") or {}
    task_type = task.get("task_type")
    target_tool = _target_tool(task)
    relevant = expected.get("relevant_memory")
    relevant_tool = relevant.get("tool_call") if isinstance(relevant, dict) else None
    ignored_tool = first_ignored_tool(expected)

    if task_type == "current_override":
        if ignored_tool:
            return ignored_tool
        if isinstance(relevant_tool, dict):
            return copy_jsonable(relevant_tool)
        if target_tool:
            return mutate_tool_call(target_tool)
        return "Reject"

    if task_type == "noise_no_inherit":
        if ignored_tool:
            return ignored_tool
        return copy_jsonable(relevant_tool) if isinstance(relevant_tool, dict) else {
            "name": "WindowControl",
            "arguments": {"action": "打开", "device": "车窗", "position": "主驾"},
        }

    if task_type == "clarify_missing":
        fallback = ignored_tool or relevant_tool or {"name": "ClimateControl", "arguments": {"action": "调到", "device": "空调", "value": "自动"}}
        return copy_jsonable(fallback)

    if task_type == "action_flip" and target_tool:
        wrong_tool = copy_jsonable(target_tool)
        args = wrong_tool.get("arguments") or {}
        if isinstance(relevant_tool, dict):
            args.update((relevant_tool.get("arguments") or {}))
        return wrong_tool

    if task_type == "position_override" and target_tool:
        wrong_tool = copy_jsonable(target_tool)
        if isinstance(relevant_tool, dict):
            old_position = (relevant_tool.get("arguments") or {}).get("position")
            if old_position:
                wrong_tool.setdefault("arguments", {})["position"] = old_position
        return wrong_tool

    if task_type == "cross_tool_interrupt" and isinstance(relevant_tool, dict):
        return ignored_tool or mutate_tool_call(relevant_tool)

    if target_tool:
        return mutate_tool_call(target_tool)
    return "Reject"


def candidate_items_from_synthetic(task: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"source": "expected", "candidate": expected_assistant_output(task)},
        {"source": "synthetic_hard_negative", "candidate": make_synthetic_rejected(task)},
    ]


def group_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row.get("id") or row.get("task_id")
        if not task_id:
            raise ValueError(f"candidate row missing id/task_id: {row}")
        item = grouped.setdefault(
            str(task_id),
            {
                "id": str(task_id),
                "task_type": row.get("task_type", ""),
                "history": row.get("history", []),
                "current_query": row.get("current_query", ""),
                "expected": row.get("expected"),
                "candidates": [],
            },
        )
        candidates = row.get("candidates")
        if isinstance(candidates, list):
            for idx, candidate in enumerate(candidates):
                item["candidates"].append({"source": f"candidate_{idx}", "candidate": candidate})
        elif "candidate" in row:
            item["candidates"].append({"source": row.get("source", "candidate"), "candidate": row["candidate"]})
        else:
            raise ValueError(f"candidate row has no candidate(s): {row}")
    return grouped


def choose_preference(
    task: dict[str, Any],
    scored: list[dict[str, Any]],
    min_chosen_score: float,
    max_rejected_score: float,
    min_gap: float,
) -> dict[str, Any] | None:
    decision = choose_preference_with_reason(task, scored, min_chosen_score, max_rejected_score, min_gap)
    return decision["preference"]


def choose_preference_with_reason(
    task: dict[str, Any],
    scored: list[dict[str, Any]],
    min_chosen_score: float,
    max_rejected_score: float,
    min_gap: float,
) -> dict[str, Any]:
    ranked = sorted(scored, key=lambda item: item["verdict"]["score"], reverse=True)
    chosen = ranked[0]
    rejected = ranked[-1]
    chosen_score = float(chosen["verdict"]["score"])
    rejected_score = float(rejected["verdict"]["score"])
    score_gap = chosen_score - rejected_score
    stats = {
        "best_score": chosen_score,
        "worst_score": rejected_score,
        "score_gap": score_gap,
        "best_source": chosen.get("source", ""),
        "worst_source": rejected.get("source", ""),
    }
    if chosen_score < min_chosen_score:
        return {"preference": None, "skip_reason": "chosen_score_below_threshold", **stats}
    if rejected_score > max_rejected_score:
        return {"preference": None, "skip_reason": "rejected_score_above_threshold", **stats}
    if score_gap < min_gap:
        return {"preference": None, "skip_reason": "score_gap_below_threshold", **stats}
    if compact_json(chosen["candidate"]) == compact_json(rejected["candidate"]):
        return {"preference": None, "skip_reason": "chosen_rejected_identical", **stats}
    preference = {
        "id": task.get("id", ""),
        "task_type": task.get("task_type", ""),
        "history": task.get("history", []),
        "current_query": task.get("current_query", ""),
        "chosen": chosen["candidate"],
        "rejected": rejected["candidate"],
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
        "chosen_verdict": chosen["verdict"],
        "rejected_verdict": rejected["verdict"],
        "chosen_source": chosen.get("source", ""),
        "rejected_source": rejected.get("source", ""),
    }
    return {"preference": preference, "skip_reason": "", **stats}


def candidate_items_for_task(
    task: dict[str, Any],
    candidate_groups: dict[str, dict[str, Any]] | None,
    include_synthetic_negative: bool = True,
) -> list[dict[str, Any]]:
    if candidate_groups is not None:
        task_id = str(task.get("id", ""))
        grouped = candidate_groups.get(task_id)
        if not grouped:
            items: list[dict[str, Any]] = []
        else:
            items = list(grouped["candidates"])
        if include_synthetic_negative:
            items.append({"source": "synthetic_hard_negative", "candidate": make_synthetic_rejected(task)})
        return items
    return candidate_items_from_synthetic(task)


def process_task(
    index: int,
    total: int,
    task: dict[str, Any],
    candidate_items: list[dict[str, Any]],
    verifier_prompt: str,
    args: argparse.Namespace,
    api_key: str,
    client: Any,
) -> tuple[int, dict[str, Any] | None, dict[str, Any], str]:
    if not candidate_items:
        audit = {
            "id": task.get("id", ""),
            "task_type": task.get("task_type", ""),
            "scored": [],
            "kept": False,
            "skip_reason": "no_candidates",
        }
        return index, None, audit, f"[skip] {task.get('id', '')}: no candidates"

    scored = []
    for item in candidate_items:
        verdict = score_candidate(verifier_prompt, task, item["candidate"], args, api_key, client)
        scored.append({**item, "verdict": verdict})

    decision = choose_preference_with_reason(
        task,
        scored,
        args.min_chosen_score,
        args.max_rejected_score,
        args.min_gap,
    )
    pref = decision["preference"]
    audit = {
        "id": task.get("id", ""),
        "task_type": task.get("task_type", ""),
        "scored": scored,
        "kept": pref is not None,
        "skip_reason": decision["skip_reason"],
        "best_score": decision["best_score"],
        "worst_score": decision["worst_score"],
        "score_gap": decision["score_gap"],
        "best_source": decision["best_source"],
        "worst_source": decision["worst_source"],
    }
    msg = f"[{index}/{total}] {task.get('id', '')} scored={len(scored)} kept={pref is not None}"
    return index, pref, audit, msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build memory chosen/rejected preferences with Qwen Plus verifier.")
    p.add_argument("--tasks", default="data/rl/memory_tasks.jsonl")
    p.add_argument("--candidate-file", default="", help="Optional model candidate JSONL. If omitted, synthetic candidates are used.")
    p.add_argument("--verifier-prompt", default="data/rl/memory_verifier_prompt.md")
    p.add_argument("--output", default="data/rl/memory_preferences.jsonl")
    p.add_argument("--audit-output", default="data/rl/memory_preferences_audit.jsonl")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    p.add_argument("--api-key", default="", help="Verifier API key. Prefer environment variables to avoid shell history leaks.")
    p.add_argument("--appid-env", default=DEFAULT_APPID_ENV)
    p.add_argument("--appid", default="", help="Optional AI Hub app id. Prefer --appid-env in shared environments.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-chosen-score", type=float, default=8.0)
    p.add_argument("--max-rejected-score", type=float, default=4.0)
    p.add_argument("--min-gap", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=1, help="Concurrent verifier tasks. Start with 4; reduce if rate-limited.")
    p.add_argument(
        "--no-synthetic-negative",
        action="store_true",
        help="When --candidate-file is used, do not mix in a synthetic hard negative.",
    )
    p.add_argument(
        "--append-output",
        action="store_true",
        help="Append preference/audit rows instead of overwriting output files. Use carefully to avoid duplicates.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = (args.api_key or os.environ.get(args.api_key_env, "")).strip()
    if not api_key:
        raise SystemExit(f"Missing API key. Export {args.api_key_env}=<your verifier key> first.")
    appid = (args.appid or os.environ.get(args.appid_env, "")).strip()
    client = create_openai_client(api_key=api_key, base_url=args.base_url, appid=appid, timeout=args.timeout)

    verifier_prompt = Path(args.verifier_prompt).read_text(encoding="utf-8")
    tasks = load_jsonl(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    candidate_groups = None
    if args.candidate_file:
        candidate_groups = group_candidate_rows(load_jsonl(args.candidate_file))

    indexed_tasks = [
        (
            idx,
            task,
            candidate_items_for_task(
                task,
                candidate_groups,
                include_synthetic_negative=not args.no_synthetic_negative,
            ),
        )
        for idx, task in enumerate(tasks, start=1)
    ]
    results: list[tuple[int, dict[str, Any] | None, dict[str, Any]]] = []

    if args.workers <= 1:
        for index, task, candidate_items in indexed_tasks:
            idx, pref, audit, msg = process_task(
                index, len(tasks), task, candidate_items, verifier_prompt, args, api_key, client
            )
            print(msg, flush=True)
            results.append((idx, pref, audit))
    else:
        print(f"[parallel] verifier workers={args.workers}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_task,
                    index,
                    len(tasks),
                    task,
                    candidate_items,
                    verifier_prompt,
                    args,
                    api_key,
                    client,
                )
                for index, task, candidate_items in indexed_tasks
            ]
            for future in as_completed(futures):
                idx, pref, audit, msg = future.result()
                print(msg, flush=True)
                results.append((idx, pref, audit))

    results.sort(key=lambda item: item[0])
    preferences = [pref for _, pref, _ in results if pref is not None]
    audit_rows = [audit for _, _, audit in results]

    writer = append_jsonl if args.append_output else write_jsonl
    writer(preferences, args.output)
    writer(audit_rows, args.audit_output)
    print(f"[out] preferences={len(preferences)} -> {args.output}")
    print(f"[audit] rows={len(audit_rows)} -> {args.audit_output}")


if __name__ == "__main__":
    main()
