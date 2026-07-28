#!/usr/bin/env python3
"""从 tools.json 生成约束解码用的联合 JSON schema。

输出结构: 单工具调用对象 或 多工具调用数组(元素为任一合法调用)。
每个工具: {"name":<const>,"arguments":<inputSchema, additionalProperties=false>}

Usage:
    python scripts/build_guided_schema.py \
        --tools data/tools.json \
        --out data/guided_schema.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

NUMERIC_PATTERN = r"^\d+(\.\d+)?%?$"
NUMERIC_DESC_RE = re.compile(r"numeric|percentage|数值|百分比", re.I)
# description 未标注但 SP 特殊规则允许数值的参数
NUMERIC_OVERRIDES = {("ClimateControl", "value")}


def build_schema(tools: list[dict]) -> dict:
    calls = []
    for tool in tools:
        args_schema = copy.deepcopy(tool["inputSchema"])
        args_schema["additionalProperties"] = False
        for key, prop in args_schema.get("properties", {}).items():
            allow_numeric = (
                (tool["name"], key) in NUMERIC_OVERRIDES
                or NUMERIC_DESC_RE.search(prop.get("description", ""))
            )
            if allow_numeric and "enum" in prop:
                enum = prop.pop("enum")
                prop.pop("type", None)
                prop["anyOf"] = [
                    {"type": "string", "enum": enum},
                    {"type": "string", "pattern": NUMERIC_PATTERN},
                ]
        calls.append({
            "type": "object",
            "properties": {
                "name": {"const": tool["name"]},
                "arguments": args_schema,
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    call = {"anyOf": calls}
    return {
        "anyOf": [
            call,
            # SP 规定数组仅用于多指令,但放宽 minItems=1 以免约束把单指令硬凑成两条
            {"type": "array", "items": call, "minItems": 1},
        ]
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", default="data/tools.json")
    ap.add_argument("--out", default="data/guided_schema.json")
    args = ap.parse_args()

    tools = json.loads(Path(args.tools).read_text(encoding="utf-8"))
    schema = build_schema(tools)
    Path(args.out).write_text(
        json.dumps(schema, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"tools: {len(tools)} -> {args.out}")


if __name__ == "__main__":
    main()
