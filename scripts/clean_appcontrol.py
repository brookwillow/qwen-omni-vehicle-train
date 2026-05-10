"""
Clean data/splits/by_tool/AppControl.jsonl:
  1. Remove exact duplicate samples (same user query + same final answer)
  2. Add missing enum feature samples: 全域共驾, 语音助手
"""
import json
import shutil
from pathlib import Path

IN_FILE = Path("data/splits/by_tool/AppControl.jsonl")
BAK_FILE = IN_FILE.with_suffix(".jsonl.bak")

VALID_FEATURES = {
    "ota", "全域共驾", "媒体", "导航地图", "应用列表", "投屏助手",
    "日历", "智能影像", "用户手册", "空调", "蓝牙电话", "车控", "音乐应用", "语音助手"
}

# ---------------------------------------------------------------------------
# New samples for missing enum values
# ---------------------------------------------------------------------------
NEW_SAMPLES = [
    # 全域共驾
    {
        "messages": [
            {"role": "user", "content": "帮我打开全域共驾。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "全域共驾"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已为您打开全域共驾功能。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "我想体验全域共驾，帮我开启一下。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "全域共驾"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 已经为您开启全域共驾，请注意行驶安全。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "打开全域共驾"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "全域共驾"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已完成。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "关闭全域共驾。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "关闭", "feature": "全域共驾"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已为您关闭全域共驾。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "不用全域共驾了，帮我退出。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "关闭", "feature": "全域共驾"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，全域共驾已关闭。"},
        ]
    },
    # 语音助手
    {
        "messages": [
            {"role": "user", "content": "帮我打开语音助手设置界面。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "语音助手"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已为您打开语音助手界面。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "打开语音助手"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "语音助手"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已完成。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "调出语音助手页面。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "打开", "feature": "语音助手"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，语音助手界面已打开。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "关闭语音助手界面。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "关闭", "feature": "语音助手"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，已为您关闭语音助手界面。"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "把语音助手关了。"},
            {"role": "assistant", "content": 'Action: AppControl\nAction Input: {"action": "关闭", "feature": "语音助手"}'},
            {"role": "user", "content": 'Tool Result: {"status": "success"}'},
            {"role": "assistant", "content": "Final Answer: 好的，语音助手已关闭。"},
        ]
    },
]

# ---------------------------------------------------------------------------
# Load and deduplicate
# ---------------------------------------------------------------------------
records = [json.loads(l) for l in IN_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]

seen: set[tuple[str, str]] = set()
deduped, removed = [], []
for r in records:
    key = (r["messages"][0]["content"], r["messages"][-1]["content"])
    if key in seen:
        removed.append(r)
    else:
        seen.add(key)
        deduped.append(r)

# ---------------------------------------------------------------------------
# Combine and write
# ---------------------------------------------------------------------------
final = deduped + NEW_SAMPLES
shutil.copy(IN_FILE, BAK_FILE)
with IN_FILE.open("w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"原始: {len(records)} 条")
print(f"去重删除: {len(removed)} 条")
for r in removed:
    print(f"  - {r['messages'][0]['content']!r}")
print(f"新增样本: {len(NEW_SAMPLES)} 条 (全域共驾×5, 语音助手×5)")
print(f"最终: {len(final)} 条")
print(f"备份: {BAK_FILE}")
