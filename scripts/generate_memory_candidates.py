#!/usr/bin/env python3
"""Generate final-assistant candidates from the current SFT model service."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_SERVER = os.environ.get("QWEN_OMNI_SERVER_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("QWEN_OMNI_MODEL", "qwen-omni-lora")


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


def write_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_model_output(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass
    return s


def build_dialog_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    messages = []
    for msg in task.get("history", []) or []:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    current_query = task.get("current_query", "")
    if not isinstance(current_query, str) or not current_query.strip():
        raise ValueError(f"task {task.get('id', '')} missing current_query")
    messages.append({"role": "user", "content": current_query})
    return messages


def create_openai_client(base_url: str, api_key: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)


def request_candidate(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    _ = timeout
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = resp.choices[0]
    content = getattr(choice.message, "content", "") or ""
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    if tool_calls:
        fn = tool_calls[0].function
        return json.dumps(
            {
                "name": getattr(fn, "name", ""),
                "arguments": json.loads(getattr(fn, "arguments", "{}") or "{}"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return content


def unique_candidates(candidates: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for candidate in candidates:
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True) if not isinstance(candidate, str) else candidate
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate final assistant candidates from the current model service.")
    p.add_argument("--tasks", default="data/rl/memory_tasks.jsonl")
    p.add_argument("--output", default="data/rl/memory_candidates.jsonl")
    p.add_argument("--server", default=DEFAULT_SERVER, help="OpenAI-compatible base URL, e.g. http://host:8000/v1")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=os.environ.get("QWEN_OMNI_API_KEY", "none"))
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--retry-sleep", type=float, default=1.0)
    p.add_argument("--append", action="store_true", help="Append to output instead of replacing it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    out = Path(args.output)
    if out.exists() and not args.append:
        out.write_text("", encoding="utf-8")

    client = create_openai_client(args.server, args.api_key, args.timeout)
    for idx, task in enumerate(tasks, start=1):
        messages = build_dialog_messages(task)
        candidates = []
        raw_outputs = []
        for cand_idx in range(args.num_candidates):
            last_error: Exception | None = None
            for attempt in range(args.retries + 1):
                try:
                    raw = request_candidate(
                        client,
                        args.model,
                        messages,
                        args.max_tokens,
                        args.temperature,
                        args.timeout,
                    )
                    raw_outputs.append(raw)
                    candidates.append(parse_model_output(raw))
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep * (attempt + 1))
            else:
                print(f"[warn] {task.get('id', '')} candidate={cand_idx} failed: {last_error}")

        unique = unique_candidates(candidates)
        row = {
            "id": task.get("id", ""),
            "task_type": task.get("task_type", ""),
            "history": task.get("history", []),
            "current_query": task.get("current_query", ""),
            "expected": task.get("expected"),
            "candidates": unique,
            "raw_outputs": raw_outputs,
            "messages": messages,
        }
        write_jsonl_row(out, row)
        print(f"[{idx}/{len(tasks)}] {task.get('id', '')} candidates={len(candidates)} unique={len(unique)}")

    print(f"[out] candidates -> {args.output}")


if __name__ == "__main__":
    main()
