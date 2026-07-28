#!/usr/bin/env python3
"""从现有 system-prompt.txt 生成枚举宏压缩版 SP。

策略(配合推理侧 schema 约束解码使用):
1. position 全部替换为 @P=全局位置并集,单个工具的合法子集由约束解码保证;
2. 枚举项按全局频次规范排序,使跨工具相同集合的枚举串完全一致;
3. 整串重复的枚举 -> 宏; 高频连续子串(如档位/开关块) -> 子串宏。

Usage:
    python scripts/build_macro_sp.py \
        --sp-file data/system-prompt.txt \
        --out data/system-prompt-macro.txt
"""

from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path

TOOL_LINE_RE = re.compile(r"^(\d+)\.(\w+)\((.*)\)//(.*)$")
TOOLS_HEADER = "工具(*为required):"
MIN_OCCURRENCES = 2
MIN_SAVING_CHARS = 8
# 规范排序后仍以连续子串形式出现的高频块
SUBSTRING_BLOCKS = [
    "最低|低|中|高|最高",
    "关闭|打开|调到|调高|调低",
    "关闭|打开",
]


def parse_params(param_str: str):
    parts = []
    if not param_str:
        return parts
    for seg in param_str.split(","):
        key, _, enum = seg.partition(":")
        parts.append((key, enum))
    return parts


def macro_names(used: set[str]):
    for ch in string.ascii_uppercase:
        if ch not in used:
            yield ch
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield a + b


def build_macro_sp(text: str) -> tuple[str, dict[str, str]]:
    """压缩 SP 文本,返回 (压缩后文本, {宏名: 展开串})。"""
    lines = text.rstrip("\n").split("\n")

    header_end = next(i for i, l in enumerate(lines) if l.strip() == TOOLS_HEADER)
    header = lines[:header_end]
    tool_lines, footer = [], []
    for line in lines[header_end + 1:]:
        if TOOL_LINE_RE.match(line):
            tool_lines.append(line)
        else:
            footer.append(line)

    parsed = []
    item_freq: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for line in tool_lines:
        num, name, param_str, comment = TOOL_LINE_RE.match(line).groups()
        params = parse_params(param_str)
        parsed.append((num, name, params, comment))
        for _, enum in params:
            for it in enum.split("|"):
                item_freq[it] += 1
                first_seen.setdefault(it, len(first_seen))

    def canon(enum: str) -> str:
        items = enum.split("|")
        # 数值/占位符(<数字>等)保持在尾部原序,其余按全局频次排序
        return "|".join(sorted(items, key=lambda it: (-item_freq[it], first_seen[it])))

    # @P: 全局 position 并集(按频次排序,合法子集由约束解码保证)
    pos_items: list[str] = []
    for _, _, params, _ in parsed:
        for key, enum in params:
            if key.lstrip("*") == "position":
                for it in enum.split("|"):
                    if it not in pos_items:
                        pos_items.append(it)
    pos_union = canon("|".join(pos_items))

    macros: dict[str, str] = {pos_union: "@P"}

    # 整串宏候选: 规范排序后完全相同且总节省达阈值的枚举串
    enum_counter: Counter[str] = Counter()
    for _, _, params, _ in parsed:
        for key, enum in params:
            if key.lstrip("*") == "position":
                continue
            if "|" in enum:
                enum_counter[canon(enum)] += 1
    candidates = sorted(
        ((e, c) for e, c in enum_counter.items()
         if c >= MIN_OCCURRENCES and (c - 1) * len(e) >= MIN_SAVING_CHARS),
        key=lambda x: (x[1] - 1) * len(x[0]), reverse=True,
    )
    names = macro_names(used={"P"})
    for enum, _ in candidates:
        macros[enum] = f"@{next(names)}"

    def replace_block(enum: str, blk: str, mname: str) -> str:
        # 仅在 | 边界处整块替换,避免跨枚举项误匹配
        items, blk_items = enum.split("|"), blk.split("|")
        out, i = [], 0
        while i < len(items):
            if items[i:i + len(blk_items)] == blk_items:
                out.append(mname)
                i += len(blk_items)
            else:
                out.append(items[i])
                i += 1
        return "|".join(out)

    # 子串宏: 剩余枚举串内的高频连续块(与整串宏同串时复用宏名)
    sub_macros: dict[str, str] = {}
    for block in SUBSTRING_BLOCKS:
        blk = canon(block)
        occurs = 0
        for _, _, params, _ in parsed:
            for key, enum in params:
                if key.lstrip("*") == "position":
                    continue
                ce = canon(enum)
                if ce not in macros and ce != blk and replace_block(ce, blk, "@") != ce:
                    occurs += 1
        if occurs >= MIN_OCCURRENCES and (occurs - 1) * len(blk) >= MIN_SAVING_CHARS:
            sub_macros[blk] = macros.get(blk) or f"@{next(names)}"

    def render_enum(key: str, enum: str) -> str:
        if key.lstrip("*") == "position":
            return "@P"
        if "|" not in enum:
            return enum
        ce = canon(enum)
        if ce in macros:
            return macros[ce]
        for blk, mname in sub_macros.items():
            ce = replace_block(ce, blk, mname)
        return ce

    out_tool_lines = []
    for num, name, params, comment in parsed:
        segs = [f"{key}:{render_enum(key, enum)}" for key, enum in params]
        out_tool_lines.append(f"{num}.{name}({','.join(segs)})//{comment}")

    all_macros: dict[str, str] = {}
    for e, m in [*macros.items(), *sub_macros.items()]:
        all_macros.setdefault(m, e)
    vocab_line = "词表(@X按此展开,@P为各工具position可选范围的并集,以实际支持为准): " + " ".join(
        f"{m}={e}" for m, e in all_macros.items()
    )

    out_lines = header + [TOOLS_HEADER, vocab_line] + out_tool_lines + footer
    return "\n".join(out_lines) + "\n", all_macros


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp-file", default="data/system-prompt.txt")
    ap.add_argument("--out", default="data/system-prompt-macro.txt")
    args = ap.parse_args()

    text = Path(args.sp_file).read_text(encoding="utf-8")
    out_text, all_macros = build_macro_sp(text)
    Path(args.out).write_text(out_text, encoding="utf-8")

    before, after = len(text), len(out_text)
    print(f"macros: {len(all_macros)}")
    for m, enum in all_macros.items():
        print(f"  {m}: {enum[:70]}{'...' if len(enum) > 70 else ''}")
    print(f"chars: {before} -> {after} ({(before - after) / before:.1%} saved)")

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        tb, ta = len(tok(text).input_ids), len(tok(out_text).input_ids)
        print(f"tokens(qwen): {tb} -> {ta} ({(tb - ta) / tb:.1%} saved)")
    except Exception as e:
        print(f"[skip token count] {e}")


if __name__ == "__main__":
    main()
