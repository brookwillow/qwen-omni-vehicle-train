#!/usr/bin/env python3
"""
rewrite_text_context.py
用 Claude API 改写 text_context.jsonl 中无意义的 assistant 中间轮回复，
使其符合车载语音助手风格（有实质内容、自然流畅、简短不啰嗦）。
"""
import json
import re
import time
import argparse
from pathlib import Path
import anthropic

# 无意义回复的关键词列表，命中则需改写
MEANINGLESS = {"好的。", "明白。", "了解。", "是的。", "嗯。", "可以。",
               "听起来不错。", "不错。", "可以试试。", "我在，请说。"}

SYSTEM_PROMPT = """你是一个车载语音助手对话数据改写专家。

任务：改写多轮对话中无意义的 assistant 中间轮回复，使其符合车载语音助手的风格。

要求：
1. 回复简短自然（10~25字），符合车载场景语音交互
2. 必须结合上下文，给出有实质意义的回应（理解用户意图、提示下一步、或确认当前状态）
3. 不要执行工具调用，只输出自然语言文本
4. 保持友好、专业的语音助手口吻
5. 只输出改写后的回复文本，不要任何解释

改写示例：
- 上文：用户说"车里有点热"，原回复"好的。" → 改写："好的，我来帮您调低空调温度。"
- 上文：用户说"刚才谢谢你"，原回复"不客气。" → 改写："不客气，有需要随时告诉我。"
- 上文：用户说"等等"，原回复"好的。" → 改写："好的，您慢慢说，我随时听候。"
- 上文：用户说"百科说现在很冷"，原回复"嗯。" → 改写："嗯，需要我帮您把空调温度调高一些吗？"
"""


def is_meaningless(text: str) -> bool:
    text = text.strip()
    if text.startswith("{") or text == "Reject":
        return False
    if text in MEANINGLESS:
        return True
    # 5字以内且不含功能性词汇
    if len(text) <= 5 and not any(k in text for k in ["帮", "调", "打开", "关闭", "查", "切"]):
        return True
    return False


def build_rewrite_prompt(messages: list, target_idx: int) -> str:
    """构建改写 prompt，提供上下文"""
    lines = ["对话上下文："]
    # 提供目标 turn 前后各2轮作为上下文
    start = max(0, target_idx - 2)
    end = min(len(messages), target_idx + 2)
    for i, m in enumerate(messages[start:end], start):
        role = "用户" if m["role"] == "user" else "助手"
        content = m["content"] if not m["content"].startswith("{") else "[工具调用]"
        marker = " ← 需要改写" if i == target_idx else ""
        lines.append(f"  {role}: {content}{marker}")

    lines.append(f"\n需要改写的原始回复：「{messages[target_idx]['content']}」")
    lines.append("\n请直接输出改写后的回复（只输出文本，不要解释）：")
    return "\n".join(lines)


def rewrite_batch(client: anthropic.Anthropic, samples: list, dry_run: bool = False):
    rewritten_count = 0
    total_targets = 0

    for sample_idx, sample in enumerate(samples):
        msgs = sample["messages"]
        changed = False

        for i, m in enumerate(msgs):
            # 只改写非最后一轮的 assistant 回复
            if m["role"] != "assistant":
                continue
            if i == len(msgs) - 1:
                continue
            if not is_meaningless(m["content"]):
                continue

            total_targets += 1
            original = m["content"]

            if dry_run:
                print(f"[dry-run] 样本{sample_idx} turn{i}: 「{original}」")
                continue

            prompt = build_rewrite_prompt(msgs, i)
            try:
                resp = client.messages.create(
                    model="claude-haiku-4.5",
                    max_tokens=64,
                    messages=[
                        {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}
                    ],
                )
                new_text = resp.content[0].text.strip().strip("「」")
                m["content"] = new_text
                changed = True
                rewritten_count += 1
                print(f"[{sample_idx:03d}|turn{i}] 「{original}」→ 「{new_text}」")
                time.sleep(0.1)  # 避免限速
            except Exception as e:
                print(f"[ERROR] 样本{sample_idx} turn{i}: {e}")

    return samples, rewritten_count, total_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/splits/text_context.jsonl")
    parser.add_argument("--output", default="data/splits/text_context.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不改写")
    args = parser.parse_args()

    samples = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"加载 {len(samples)} 条样本")

    client = anthropic.Anthropic()

    samples, rewritten, total = rewrite_batch(client, samples, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] 需要改写的 turn 数: {total}")
        return

    # 备份原文件
    bak = Path(args.input).with_suffix(".jsonl.bak")
    if not bak.exists():
        import shutil
        shutil.copy(args.input, bak)
        print(f"已备份原文件 → {bak}")

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n完成：改写 {rewritten}/{total} 个 turn → {args.output}")


if __name__ == "__main__":
    main()
