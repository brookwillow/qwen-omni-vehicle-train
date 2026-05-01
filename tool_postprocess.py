"""Post-processing for parsed tool calls."""

from __future__ import annotations

import re
from typing import Any


SEAT_POSITION_TERMS = (
    "主驾",
    "驾驶位",
    "司机位",
    "副驾",
    "副驾驶",
    "前排",
    "后排",
    "第一排",
    "第二排",
    "第三排",
    "左侧",
    "右侧",
    "中间",
)


def _mentions_seat_position(query: str) -> bool:
    return any(term in query for term in SEAT_POSITION_TERMS)


def _infer_window_position(query: str) -> str | None:
    if any(term in query for term in ("所有", "全部", "都")):
        return "全部"
    if "主驾" in query:
        return "主驾"
    if "副驾" in query or "副驾驶" in query:
        return "副驾"
    if "前排" in query:
        return "前排"
    if "左边" in query or "左侧" in query:
        return "左侧"
    if "右边" in query or "右侧" in query:
        return "右侧"
    if "第二排" in query:
        return "第二排"
    if "第三排" in query:
        return "第三排"
    return None


def _infer_empty_window_args(query: str) -> dict[str, Any]:
    if not any(term in query for term in ("车窗", "窗户", "窗")):
        return {}

    first_part = re.split(r"[，,。；;]|顺便|然后|再把|也", query, maxsplit=1)[0]
    fixed: dict[str, Any] = {"device": "车窗"}
    position = _infer_window_position(first_part)
    if position:
        fixed["position"] = position

    percent_match = re.search(r"(\d{1,3})\s*%", first_part)
    if percent_match:
        fixed["action"] = "开到" if any(term in first_part for term in ("开", "打开", "摇下")) else "关到"
        fixed["value"] = f"{percent_match.group(1)}%"
    elif "一半" in first_part or "半" in first_part:
        fixed["action"] = "开到" if any(term in first_part for term in ("开", "打开", "摇下")) else "关到"
        fixed["value"] = "50%"
    elif "一条缝" in first_part:
        fixed["action"] = "开到"
        fixed["value"] = "10%"
    elif any(term in first_part for term in ("关闭", "关上", "关了", "关掉")):
        fixed["action"] = "关闭"
    elif any(term in first_part for term in ("打开", "开窗", "开一下", "开了", "摇下来", "通风")):
        fixed["action"] = "打开"

    if "action" not in fixed:
        return {}
    return fixed


def postprocess_action_args(query: str, tool: str | None, args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fix systematic parser/model artifacts without changing explicit user intent."""
    if not isinstance(args, dict):
        return args

    fixed = dict(args)
    query = query or ""
    if tool == "SeatControl":
        if fixed.get("position") == "主驾" and not _mentions_seat_position(query):
            fixed.pop("position", None)
        if fixed.get("device") == "座椅" and fixed.get("action") in {"调前", "调后"}:
            fixed.setdefault("feature", "位置")
        if fixed.get("action") == "切换" and fixed.get("value"):
            fixed["action"] = "调到"
    elif tool == "WindowControl":
        if not fixed:
            fixed = _infer_empty_window_args(query)
        value = fixed.get("value")
        if isinstance(value, str) and value.isdigit() and f"{value}%" in query:
            fixed["value"] = f"{value}%"
    return fixed
