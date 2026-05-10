"""
Append generalized/paraphrased CameraControl samples to
`data/splits/by_tool/CameraControl.jsonl`.
Behavior:
 - Load existing records
 - Generate templated variations for `action`, `device`, `position`, `value`
 - Avoid destination-like content (not relevant for cameras)
 - Deduplicate exact (user + final answer) duplicates
 - Backup original to .bak and write new file
"""
import json
import random
import shutil
from pathlib import Path

IN_FILE = Path('data/splits/by_tool/CameraControl.jsonl')
BAK_FILE = IN_FILE.with_suffix('.jsonl.bak')

ACTIONS = ['打开','关闭']
DEVICES = ['摄像头','舱内摄像头']
POSITIONS = ['前侧','右侧','后侧','左侧']
VALUES = ['哨兵模式','录像','拍照','透明底盘']

OPEN_TEMPLATES = [
    '帮我把{target}打开',
    '切换到{target}界面',
    '把{target}调出来',
    '打开{target}',
    '我想看下{target}',
]
CLOSE_TEMPLATES = [
    '把{target}关掉',
    '关闭{target}',
    '退出{target}界面',
]

# mappings of targets
def target_for_value(v):
    return {
        '哨兵模式': '哨兵模式',
        '录像': '录像模式',
        '拍照': '拍照功能',
        '透明底盘': '透明底盘',
    }[v]

random.seed(1)

def gen_samples(n_each=4):
    samples = []
    # value-based commands (哨兵/录像/拍照/透明底盘)
    for v in VALUES:
        for _ in range(n_each):
            use_open = random.random() < 0.85
            target = target_for_value(v)
            if use_open:
                user = random.choice(OPEN_TEMPLATES).format(target=target)
                action = '打开'
            else:
                user = random.choice(CLOSE_TEMPLATES).format(target=target)
                action = '关闭'
            assistant_action = f'Action: CameraControl\nAction Input: {{"action": "{action}", "device": "摄像头", "value": "{v}"}}'
            samples.append({
                'messages': [
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant_action},
                    {'role': 'user', 'content': '{"status": "success"}'},
                    {'role': 'assistant', 'content': 'Final Answer: 好的，已完成。'},
                ]
            })
    # position-based camera open/close
    for pos in POSITIONS:
        for _ in range(n_each):
            user = random.choice(OPEN_TEMPLATES).format(target=f'{pos}摄像头')
            assistant_action = f'Action: CameraControl\nAction Input: {{"action": "打开", "device": "摄像头", "position": "{pos}"}}'
            samples.append({
                'messages': [
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant_action},
                    {'role': 'user', 'content': '{"status": "success"}'},
                    {'role': 'assistant', 'content': 'Final Answer: 好的，已完成。'},
                ]
            })
    # in-cabin camera examples
    for _ in range(n_each):
        user = random.choice(OPEN_TEMPLATES).format(target='舱内摄像头')
        assistant_action = 'Action: CameraControl\nAction Input: {"action": "打开", "device": "舱内摄像头"}'
        samples.append({
            'messages': [
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant_action},
                {'role': 'user', 'content': '{"status": "success"}'},
                {'role': 'assistant', 'content': 'Final Answer: 好的，已完成。'},
            ]
        })
    return samples

if __name__ == '__main__':
    if not IN_FILE.exists():
        print('Missing input file:', IN_FILE)
        raise SystemExit(1)
    orig = [json.loads(l) for l in IN_FILE.read_text(encoding='utf-8').splitlines() if l.strip()]
    new = gen_samples(n_each=5)
    combined = orig + new
    # dedupe exact (user + final answer)
    seen = set()
    out = []
    removed = 0
    for r in combined:
        key = (r['messages'][0]['content'].strip(), r['messages'][-1]['content'].strip())
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(r)
    shutil.copy(IN_FILE, BAK_FILE)
    with IN_FILE.open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'原始 {len(orig)} 条，新增尝试 {len(new)} 条，去重后写入 {len(out)} 条，去除重复 {removed} 条，备份: {BAK_FILE}')
