"""
Extract ETA-like utterances from CarUsageSearch.jsonl and remove them from the file.
Writes:
 - data/splits/by_tool/CarUsageSearch_eta_candidates.jsonl (extracted ETA samples)
 - overwrites data/splits/by_tool/CarUsageSearch.jsonl with remaining samples
Backups original to .bak
"""
import json
import re
from pathlib import Path
import shutil

IN_FILE = Path('data/splits/by_tool/CarUsageSearch.jsonl')
BAK_FILE = IN_FILE.with_suffix('.jsonl.bak')
OUT_FILE = Path('data/splits/by_tool/CarUsageSearch_eta_candidates.jsonl')

ETA_RE = re.compile(r'(我.*久.*到|还有多久到|还有多久到达|还要多久到|多久到|多久到达|多长时间.*到|我还有多久到|还要开多远|还有多少路要走)', re.I)

lines = [l for l in IN_FILE.read_text(encoding='utf-8').splitlines() if l.strip()]
orig_count = len(lines)
eta = []
rest = []
for l in lines:
    try:
        rec = json.loads(l)
    except Exception:
        rest.append(l)
        continue
    user = ''
    if rec.get('messages') and isinstance(rec['messages'], list):
        user = rec['messages'][0].get('content','')
    if ETA_RE.search(user):
        eta.append(rec)
    else:
        rest.append(rec)

# backup
shutil.copy(IN_FILE, BAK_FILE)
# write extracted
with OUT_FILE.open('w', encoding='utf-8') as f:
    for r in eta:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
# write remaining back
with IN_FILE.open('w', encoding='utf-8') as f:
    for r in rest:
        if isinstance(r, str):
            f.write(r + '\n')
        else:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Original: {orig_count}, extracted ETA: {len(eta)}, remaining: {len(rest)}')
print(f'ETA candidates written to: {OUT_FILE}')
print(f'Backup saved to: {BAK_FILE}')
