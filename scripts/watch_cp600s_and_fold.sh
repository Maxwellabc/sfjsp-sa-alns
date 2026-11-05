#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/fjsp-combo-b
RAW=$ROOT/results/p0-cp-stoch-600s-raw.csv
LOG=$ROOT/results/p0-cp-stoch-600s-watch.log
DONE=$ROOT/results/p0-cp-stoch-600s.DONE
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T %Z')] watch start"

rows() { if [[ -f "$RAW" ]]; then echo $(($(wc -l <"$RAW")-1)); else echo 0; fi; }

CP_PID=$(pgrep -f 'run_cp_stoch_p0.py --time-limit 600 --out-prefix p0-cp-stoch-600s' | head -1 || true)
if [[ -n "${CP_PID:-}" ]]; then
  echo "attach pid=$CP_PID rows=$(rows)"
  while kill -0 "$CP_PID" 2>/dev/null; do
    echo "[$(date '+%F %T')] pid=$CP_PID rows=$(rows)"
    [[ "$(rows)" -ge 20 ]] && break
    sleep 120
  done
fi

if [[ "$(rows)" -lt 20 ]]; then
  echo "resume remaining from rows=$(rows)"
  cd "$ROOT"
  PYTHONPATH=src .venv/bin/python -u scripts/run_cp_stoch_p0.py \
    --time-limit 600 --out-prefix p0-cp-stoch-600s --workers 2 --resume \
    >>"$ROOT/results/p0-cp-stoch-600s-run.log" 2>&1
fi

echo "final rows=$(rows)"
[[ "$(rows)" -ge 20 ]] || { echo INCOMPLETE; exit 2; }

cd "$ROOT"
PYTHONPATH=src .venv/bin/python - <<'PY'
import csv, json, time
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/fjsp-combo-b')
rows = list(csv.DictReader((ROOT/'results/p0-cp-stoch-600s-raw.csv').open()))
for r in rows:
    for k in ('cp_obj_search','cp_wall_s','eval_mean_cmax','eval_var95','time_limit_s'):
        if r.get(k) not in (None,''):
            r[k] = float(r[k])
    for k in ('solution_count','num_realizations_cp','workers'):
        if r.get(k) not in (None,''):
            r[k] = int(float(r[k]))
feasible = [r for r in rows if r.get('status')=='FEASIBLE' and r.get('eval_mean_cmax') is not None]
summary = {
  'protocol': {
    'instances': len(rows), 'time_limit_s': 600, 'num_realizations_cp': 10, 'obj': 'mean',
    'eval_slice': [250, 350],
    'note': 'CP-stoch official mean; K_cp=10; medium budget 600s; held-out [250:350]. Not paper 3600s.',
  },
  'n_feasible': len(feasible),
  'n_failed': len(rows)-len(feasible),
  'macro_mean_cmax': float(np.mean([r['eval_mean_cmax'] for r in feasible])),
  'macro_mean_var95': float(np.mean([r['eval_var95'] for r in feasible])),
  'macro_cp_obj_search': float(np.mean([r['cp_obj_search'] for r in feasible])),
  'mean_wall_s': float(np.mean([r['cp_wall_s'] for r in feasible])),
  'per_instance': rows,
}
(ROOT/'results/p0-cp-stoch-600s-summary.json').write_text(json.dumps(summary, indent=2))
lines = [
  '# P0 CP-stoch (medium budget 600s)',
  '',
  f"- Feasible {len(feasible)}/{len(rows)}; Macro E={summary['macro_mean_cmax']:.4f}; VaR95={summary['macro_mean_var95']:.4f}",
  f"- Mean wall {summary['mean_wall_s']:.1f}s; search obj {summary['macro_cp_obj_search']:.4f}",
  '',
  '| Instance | status | search | eval E | VaR95 | wall |',
  '|---|---|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(f"| {r['instance']} | {r['status']} | {r.get('cp_obj_search')} | {r.get('eval_mean_cmax')} | {r.get('eval_var95')} | {r.get('cp_wall_s')} |")
(ROOT/'results/p0-cp-stoch-600s.md').write_text('\n'.join(lines)+'\n')
cp60 = json.loads((ROOT/'results/p0-cp-stoch-summary.json').read_text())
cp60_e, cp60_v = float(cp60['macro_mean_cmax']), float(cp60['macro_mean_var95'])
cp600_e, cp600_v = summary['macro_mean_cmax'], summary['macro_mean_var95']

def gap(a,b):
    return 100.0*(a-b)/b

out = {
  'macro': {
    'MOR': {'E': 654.31, 'VaR95': 762.64},
    'SA-ALNS': {'E': 597.16, 'VaR95': 683.98},
    'CP-stoch-60s': {'E': cp60_e, 'VaR95': cp60_v},
    'CP-stoch-600s': {'E': cp600_e, 'VaR95': cp600_v},
    'SPM-DAN': 'N/A',
  },
  'alns_vs_cp600_pct': gap(597.16, cp600_e),
  'notes': ['P0 20-inst MOR/ALNS; CP-600s medium not 3600s; full-100 separate bank'],
}
(ROOT/'results/p0-baselines-compare-summary.json').write_text(json.dumps(out, indent=2))
(ROOT/'results/p0-baselines-compare.md').write_text('\n'.join([
  '# P0 baselines comparison (MOR / SA-ALNS / CP-60s / CP-600s / SPM-DAN)',
  '',
  '| Method | E[Cmax] | VaR95 | vs MOR % | note |',
  '|---|---:|---:|---:|---|',
  f"| MOR | 654.31 | 762.64 | — | P0 20× |",
  f"| SA-ALNS | 597.16 | 683.98 | {gap(597.16,654.31):.2f} | P0 20× |",
  f"| CP-stoch (60s) | {cp60_e:.2f} | {cp60_v:.2f} | {gap(cp60_e,654.31):.2f} | short |",
  f"| CP-stoch (600s) | {cp600_e:.2f} | {cp600_v:.2f} | {gap(cp600_e,654.31):.2f} | medium; not 3600s |",
  '| SPM-DAN | N/A | N/A | — | no public FJSP weights |',
  '',
  f"ALNS vs CP-600s: {gap(597.16, cp600_e):.2f}% (negative => ALNS better).",
  '',
])+'\n')
# light touch draft-experiments
exp = ROOT/'docs/draft-experiments.md'
if exp.exists():
    txt = exp.read_text()
    banner = f"\n\n## CP-stoch 600s (auto-fold {time.strftime('%Y-%m-%d %H:%M')})\nMacro E={cp600_e:.4f}, VaR95={cp600_v:.4f}, n={len(feasible)}. Medium budget; not 3600s.\n"
    if 'CP-stoch 600s (auto-fold' not in txt:
        exp.write_text(txt + banner)
marker = {'macro_mean_cmax': cp600_e, 'macro_mean_var95': cp600_v, 'n': len(feasible), 'ts': time.time()}
(ROOT/'results/p0-cp-stoch-600s.DONE').write_text(json.dumps(marker, indent=2))
print(json.dumps(marker, indent=2))
PY

echo "[$(date '+%F %T %Z')] fold complete"
