# Reproduce this table from scratch

This table was assembled from `eval_summary.json` files under:

- `/vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo`

## Commands used

```bash
# 1) Find all eval_summary files
find /vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo -name eval_summary.json

# 2) Read each json and extract fields used below
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

root = Path('/vol/bitbucket/aa6622/project/artifacts/phase11_env_on_policy_grpo')
rows = []
for f in sorted(root.glob('**/eval_summary.json')):
    d = json.loads(f.read_text())
    rows.append({
        'run_dir': f.parent.name,
        'ckpt': Path(d['grpo_checkpoint']).name,
        'seed': int(d['eval_seed_start']),
        'episodes': int(d['episodes']),
        'pc_success': float(d['pc_success']),
    })

# 3) Group by checkpoint filename (update_XXXX.pt)
g = defaultdict(dict)
for r in rows:
    if r['ckpt'].startswith('update_'):
        g[r['ckpt']][r['seed']] = r

# 4) For each checkpoint, map seed groups and compute overall
for ckpt in sorted(g):
    a = g[ckpt].get(1000)  # usually 20-episode runs
    b = g[ckpt].get(1020)  # usually 30-episode runs
    overall = None
    if a and b:
        # weighted by episode count: 20+30=50
        overall = (a['pc_success'] * 20 + b['pc_success'] * 30) / 50
    elif a:
        overall = a['pc_success']
    elif b:
        overall = b['pc_success']
    print(ckpt, a['run_dir'] if a else '-', a['pc_success'] if a else None,
          b['run_dir'] if b else '-', b['pc_success'] if b else None, overall)
PY
```

## Why these values are correct

- `pc_success` is already stored as percentage.
- Extra eval directories are also included when they map to known checkpoints:
  - `eval_grpo_u1_20ep` → `update_0001.pt`
  - `eval_grpo_u5_20ep` → `update_0005.pt`
  - `eval_smoke_v2` → `update_0001.pt` (2-episode smoke run)
- The LaTeX table shows combined checkpoint-level view with a weighted overall when both seed blocks exist.
