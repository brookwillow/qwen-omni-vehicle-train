"""
Strip trailing punctuation from the last character of user messages in JSONL files.
Targets (default): recursive under `data/eval` and `data/splits`.
Creates a `.bak` copy for each file before writing.

Usage:
  python3 scripts/strip_final_punctuation.py
"""
from pathlib import Path
import json
import shutil

ROOTS = [Path('data/eval'), Path('data/splits')]
PUNCT = set(list('。？，、；:：!?.,;!'))

def strip_trailing(s: str) -> str:
    s = s.rstrip()
    if not s: return s
    # If last char is punctuation, remove it
    if s[-1] in PUNCT:
        return s[:-1]
    return s

def process_file(p: Path):
    lines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    changed = 0
    out_lines = []
    for l in lines:
        try:
            obj = json.loads(l)
        except Exception:
            out_lines.append(l)
            continue
        # find first user role message(s) and strip trailing punctuation
        msgs = obj.get('messages') or obj.get('dialog') or None
        if isinstance(msgs, list):
            modified = False
            for m in msgs:
                if m.get('role') == 'user' and isinstance(m.get('content'), str):
                    newc = strip_trailing(m['content'])
                    if newc != m['content']:
                        m['content'] = newc
                        modified = True
            if modified:
                changed += 1
        out_lines.append(json.dumps(obj, ensure_ascii=False))
    if changed:
        bak = p.with_suffix(p.suffix + '.bak')
        shutil.copy(p, bak)
        p.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return len(lines), changed

if __name__ == '__main__':
    total_files = 0
    total_lines = 0
    total_changed = 0
    files = []
    for root in ROOTS:
        if not root.exists():
            continue
        files += list(root.rglob('*.jsonl'))
    for f in sorted(files):
        total_files += 1
        lines, changed = process_file(f)
        total_lines += lines
        total_changed += changed
        print(f"Processed {f}: lines={lines}, modified_records={changed}")
    print('---')
    print(f"Files scanned: {total_files}")
    print(f"Total records: {total_lines}")
    print(f"Total modified records: {total_changed}")
