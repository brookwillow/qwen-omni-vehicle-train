"""
Generate generalized/paraphrased AppControl samples and append to
`data/splits/by_tool/AppControl.jsonl` (with backup + dedupe).

Behavior:
 - Load existing records
 - Generate templated variations for each enum feature (avoid navigation destinations)
 - Deduplicate exact (user + final answer) duplicates
 - Backup original file to .bak and write new file
"""
import json
import random
from pathlib import Path
import shutil

IN_PATH = Path('data/splits/by_tool/AppControl.jsonl')
BAK_PATH = IN_PATH.with_suffix('.jsonl.bak')

# Enum features from tools.json (kept in sync)
FEATURES = [
    "ota","全域共驾","媒体","导航地图","应用列表","投屏助手",
    "日历","智能影像","用户手册","空调","蓝牙电话","车控","音乐应用","语音助手"
]

# Generic templates that insert the feature name; keep them safe (no destination navigation)
OPEN_TEMPLATES = [
    "请把 {f} 打开一下",
    "帮我打开{f}",
    "把{f}调出来",
    "切换到{f}界面",
    "显示{f}",
    "调出{f}页面",
    "给我打开{f}",
    "我要用{f}",
    "帮我进入{f}",
    "把{f}打开",
]
CLOSE_TEMPLATES = [
    "把{f}关掉",
    "关闭{f}",
    "退出{f}界面",
    "把{f}收起来",
]

# Some feature-specific short utterances (mapping feature -> extras)
EXTRA_UTTER = {
    "音乐应用": ["放点歌", "来点音乐", "我想听歌"],
    "导航地图": ["打开地图", "看下地图", "调出地图"],
    "日历": ["看一下今天的日程", "帮我看下日程", "打开日历看看"],
    "蓝牙电话": ["我要打电话", "打开电话界面", "调出通话页面"],
}

# Generate samples per feature
def gen_samples_per_feature(f, n=3):
    samples = []
    # choose a mix of open/close templates
    for _ in range(n):
        if f in ("ota",):
            tmpl = random.choice(OPEN_TEMPLATES)
            action = "打开"
        else:
            # bias to open
            if random.random() < 0.8:
                tmpl = random.choice(OPEN_TEMPLATES)
                action = "打开"
            else:
                tmpl = random.choice(CLOSE_TEMPLATES)
                action = "关闭"
        user = tmpl.format(f=f)
        # if extra utterances available, sometimes use them
        if random.random() < 0.35 and f in EXTRA_UTTER:
            user = random.choice(EXTRA_UTTER[f])
            # map to open/close heuristically
            if any(w in user for w in ("开","打开","调出","进入","显示","放")):
                action = "打开"
            if any(w in user for w in ("关","关闭","退出")):
                action = "关闭"
        assistant_action = f'Action: AppControl\nAction Input: {{"action": "{action}", "feature": "{f}"}}'
        samples.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant_action},
                {"role": "user", "content": '{"status": "success"}'},
                {"role": "assistant", "content": "Final Answer: 好的，已完成。"},
            ]
        })
    return samples

if __name__ == '__main__':
    random.seed(42)
    # load existing
    if not IN_PATH.exists():
        print("Input file missing:", IN_PATH)
        raise SystemExit(1)
    records = [json.loads(l) for l in IN_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]

    # generate 3 new variations per feature
    new_samples = []
    for f in FEATURES:
        new_samples.extend(gen_samples_per_feature(f, n=3))

    # shuffle and append
    combined = records + new_samples

    # dedupe exact (user + final answer)
    seen = set()
    out = []
    removed = 0
    for r in combined:
        user = r['messages'][0]['content']
        final = r['messages'][-1]['content']
        key = (user.strip(), final.strip())
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(r)

    # backup and write
    shutil.copy(IN_PATH, BAK_PATH)
    with IN_PATH.open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"原始 {len(records)} 条, 新增尝试 {len(new_samples)} 条, 去重后写入 {len(out)} 条, 去除重复 {removed} 条, 备份: {BAK_PATH}")
