# Scenario-aware ALNS for stochastic FJSP

Research code for Ding Junwen's ICAPS 2027 stochastic flexible job-shop (SFJSP) track: a scenario-aware adaptive large-neighborhood search (SA-ALNS) with simulated-annealing acceptance, compared with priority dispatching rules and a short-budget stochastic CP-SAT model.

The solver lives in `src/sfjsp_alns`. It does not need the NCO baseline tree for unit tests or a synthetic smoke run. Full SD3 banks and the CP model checkout are external. **SPM-DAN is N/A** — public NCO-for-Stochastic-FJSP releases do not include FJSP checkpoints, and this repo does not impute a score.

## Layout

| Path | Role |
|---|---|
| `src/sfjsp_alns/` | Instance types, `.fjs` loader, decoder, PDRs, destroy/repair, SA-ALNS |
| `scripts/alns_k_eval_1000.py` | Held-out K_eval harness (paper default 1000) |
| `scripts/alns_p0_sd3_10x5.py` | P0: first 20 × SD3 `10x5+mix`, K_eval=100, 3 seeds |
| `scripts/alns_full_sd3_10x5.py` | Full 100 × SD3 `10x5+mix`, K_eval=100, 3 seeds |
| `scripts/run_cp_stoch_p0.py` | Official CP-stoch mean model, then re-decode on the held-out slice |
| `scripts/watch_cp600s_and_fold.sh` | Resume a 600s CP job, then fold |
| `scripts/fold_cp600s_into_compare.py` | Aggregate CP-600s only from a finished raw CSV |
| `scripts/generate_sd3_10x5_realizations.py` | Lognormal random-CoV banks (needs external `.fjs` instances) |
| `data/fixtures/synthetic/` | Two tiny instances for smoke tests |
| `results/p0-cp-stoch-summary.json` | Shipped CP-stoch **60s** summary (20/20 feasible) |
| `docs/` | Paper draft and design notes |

Historical notes that mention `fjsp-combo-b/` mean this repository root.

## Setup

Python 3.11+. Core dependencies are NumPy and OR-Tools.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src
pytest
```

`pytest` does not need the SD3 tree. It loads `data/fixtures/synthetic/` and runs a 2-instance ALNS smoke inside the K_eval harness.

CP-stoch, when you actually solve, also imports the NCO benchmark package, which needs `torch` and `tomli`. Those are not required for ALNS or for `pytest`.

## Shipped measurement

JSON and markdown files under `results/` are copies from the research box. This repository bootstrap does not recompute them and does not add scores. **SPM-DAN is N/A** in every shipped summary (`spmdan` / `spmdan_status` starts with `N/A`; compare macros leave the SPM-DAN cell null).

| File | What it is |
|---|---|
| `results/p0-cp-stoch-summary.json` | CP-stoch **60s**, 20/20 feasible. Not the paper 3600s budget. |
| `results/p0-sd3-10x5-summary.json` | P0 ALNS vs PDRs, K_eval=100, 20 instances. Separate bank from the full-100 file. |
| `results/full-sd3-10x5-summary.json` | 100 × SD3 `10x5+mix`, K_eval=100. Do not merge its macros with the P0 file. |
| `results/p0-baselines-compare.md` | MOR / SA-ALNS / CP-60s. SPM-DAN row is N/A. |
| `results/k-eval-1000-summary.json` | **Partial** K_eval=1000. `batch_note` says 20/100 instances. Not the finished paper table. |
| `results/k-eval-1000-probe.json` | One-instance K_eval=1000 probe. |
| `results/k-eval-1000-summary.rows.json` | Row checkpoint for that partial batch. |

CP-60s fields in `results/p0-cp-stoch-summary.json` (copied, not recomputed here):

- Protocol: 20 × SD3 `10x5+mix`, CP-SAT stochastic **mean**, K_cp=10, **60s** (short budget), re-evaluated on scenarios `[250, 350]`.
- `n_feasible`: 20, `n_failed`: 0.
- `macro_mean_cmax`: 618.9164999999999
- `macro_mean_var95`: 722.3475
- `macro_cp_obj_search`: 573.065

Do not copy draft macros into new JSON. CP-600s artifacts are absent on purpose — `fold_cp600` writes nothing unless `results/p0-cp-stoch-600s-raw.csv` has 20 FEASIBLE rows. ALNS cells stay null unless the corresponding summary or raw CSV is present. SPM-DAN stays N/A.

## External SD3 banks

Clone the baseline and point `--data-root` at its FJSP-DRL `data` directory:

```bash
git clone https://github.com/ai-for-decision-making-tue/NCO-for-Stochastic-FJSP.git
# --data-root accepts any of:
#   <NCO>/FJSP-DRL/data
#   <NCO>/FJSP-DRL
#   <NCO>
```

Expected layout for lognormal random-CoV (no extra distribution folder):

```text
<data-root>/SD3/10x5+mix/*.fjs
<data-root>/stochastic_realizations/random/SD3/10x5+mix/*.fjs
```

Paper K_eval=1000 reads stochastic blocks `[250:1250]`, so each realizations file needs **at least 1250** blocks (`--num-samples 1250`). The P0 and full-100 scripts use K_eval=100 on `[250:350]` and are satisfied by 450 blocks. Scenario index 0 is the first block after the deterministic header, matching FJSP-DRL's loader. The nominal block is not index 0.

Generate banks once the deterministic SD3 files are on disk:

```bash
python scripts/generate_sd3_10x5_realizations.py \
  --data-root /path/to/FJSP-DRL/data \
  --size 10x5+mix \
  --limit 20 \
  --num-samples 450 \
  --seed 200

# Paper held-out slice:
python scripts/generate_sd3_10x5_realizations.py \
  --data-root /path/to/FJSP-DRL/data \
  --limit 100 \
  --num-samples 1250 \
  --seed 200
```

`--seed 200` is `seed_datagen`. The generator walks instances in sorted order on NumPy's global RNG, same recipe as `create_stochastic_realizations_random_variances` (CoV ~ U(0.1, 0.5), then lognormal). It does not import FJSP-DRL.

## K_eval harness

`scripts/alns_k_eval_1000.py` reuses `run_alns`, `best_pdr_seed`, `pdr_solution`, and `evaluate_solution`. Search and eval slices must be disjoint. Defaults are the paper split: search `[0:50]`, eval `[250:1250]`, destroy set `{random_ops, critical_path, high_var_ops, machine_load}`, repair `spt_biased`.

| Flag | Meaning |
|---|---|
| `--instances` | First N filenames (smoke: 1 or 2; full: 20 or 100) |
| `--k-eval` | Held-out scenario count (default **1000**) |
| `--seed` | Solver seed. On `--synthetic`, also the scenario-pool seed |
| `--seeds` | Comma-separated solver seeds, e.g. `50,51,52` (overrides a single `--seed` for the solver; synthetic pools still use `--seed`) |
| `--out` | JSON path. A `.csv` and `.md` are written next to it |
| `--data-root` | External FJSP-DRL data directory |
| `--synthetic` | Shipped tiny fixtures, in-process noise. **Not** an SD3 table |
| `--eval-start` / `--k-search` / `--search-start` | Slice control. Paper values are 250 / 50 / 0 |
| `--max-iter` / `--time-limit` | ALNS budget. Omitted: 40 iter / 5s on synthetic, 300 iter / 40s on an external bank |
| `--size` | SD3 folder name (default `10x5+mix`) |

If `--data-root` is omitted, the script falls back to `--synthetic` and says so on stderr. Passing both is an error. JSON field `comparable_to_paper_protocol` is true only for an external run with K_eval=1000, eval start 250, and search `[0:50]`.

### Smoke (no SD3 bank)

```bash
PYTHONPATH=src python scripts/alns_k_eval_1000.py \
  --synthetic \
  --instances 2 \
  --k-eval 8 \
  --eval-start 20 \
  --k-search 8 \
  --max-iter 15 \
  --time-limit 2 \
  --seed 50 \
  --out results/k-eval-smoke.json
```

Read the JSON. Do not put those means in the paper. `results/k-eval-smoke.*` is gitignored.

### Full K_eval=1000

Run this on the machine that has the banks. One hundred instances, three seeds, and a 40s cap is a multi-hour job. Confirm a 1-instance slice before launching the suite.

```bash
PYTHONPATH=src python scripts/alns_k_eval_1000.py \
  --data-root /path/to/FJSP-DRL/data \
  --size 10x5+mix \
  --instances 1 \
  --k-eval 1000 \
  --seed 50 \
  --max-iter 20 \
  --time-limit 15 \
  --out results/k-eval-1000-probe.json

PYTHONPATH=src python scripts/alns_k_eval_1000.py \
  --data-root /path/to/FJSP-DRL/data \
  --size 10x5+mix \
  --instances 100 \
  --k-eval 1000 \
  --seeds 50,51,52 \
  --out results/k-eval-1000-summary.json
```

The realizations files must already contain 1250 stochastic blocks. The script exits if a bank is short; it does not resample a short file up to 1000.

`results/k-eval-1000-summary.json` in this tree is the partial batch already on disk (`n_instances` 20, `batch_note` contains `20/100`). Leave it as that partial record. A finished 100-instance table replaces it only after the external run completes.

## P0 and full-100 ALNS (K_eval=100)

These match the earlier protocol: seeds 50, 51, 52; search `[0:50]`; eval `[250:350]`; 300 iterations or 40s; PDRs FIFO, MOR, SPT, MWKR plus SA-ALNS. Solver seeds are mixed with `zlib.crc32` of the filename (stable across processes).

```bash
PYTHONPATH=src python scripts/alns_p0_sd3_10x5.py \
  --data-root /path/to/FJSP-DRL/data \
  --max-inst 20

PYTHONPATH=src python scripts/alns_full_sd3_10x5.py \
  --data-root /path/to/FJSP-DRL/data \
  --max-inst 100
```

Outputs:

- `results/p0-sd3-10x5-raw.csv`, `results/p0-sd3-10x5-summary.json`, `results/p0-sd3-10x5.md`
- `results/full-sd3-10x5-raw.csv`, `results/full-sd3-10x5-summary.json`, `results/full-sd3-10x5.md`

`SFJSP_DATA_ROOT` is accepted in place of `--data-root`.

## CP-stoch

The CP model is the official `FJSPmodel_stoch` mean objective. The assignment is converted to our two-layer encoding and re-scored with `evaluate_solution` on the held-out slice (default `[250:350]`, `--eval-k 100`). Time limits:

| Invocation | Budget label | Do not call it |
|---|---|---|
| `--time-limit 60` | short | paper 3600s |
| `--time-limit 600` | medium | paper 3600s |
| `--time-limit 3600` | long | optional later; not shipped |

```bash
PYTHONPATH=src python scripts/run_cp_stoch_p0.py \
  --nco-root /path/to/NCO-for-Stochastic-FJSP \
  --data-root /path/to/FJSP-DRL/data \
  --time-limit 60 \
  --num-realizations 10 \
  --max-inst 20 \
  --workers 2 \
  --eval-start 250 \
  --eval-k 100 \
  --out-prefix p0-cp-stoch

PYTHONPATH=src python scripts/run_cp_stoch_p0.py \
  --nco-root /path/to/NCO-for-Stochastic-FJSP \
  --data-root /path/to/FJSP-DRL/data \
  --time-limit 600 \
  --num-realizations 10 \
  --max-inst 20 \
  --workers 2 \
  --out-prefix p0-cp-stoch-600s \
  --resume
```

`--resume` skips instances already marked FEASIBLE in `{out-prefix}-raw.csv`.

Two directories have to exist, because the benchmark parser joins its own `data/` tree:

```text
<nco-root>/Job_Shop_Scheduling_Benchmark_Environments_and_Instances/data/fjsp/stoch/random/SD3/10x5+mix/*.fjs
<data-root>/SD3/10x5+mix/*.fjs
<data-root>/stochastic_realizations/random/SD3/10x5+mix/*.fjs
```

The first path is what CP searches. The other two are the nominal instance and the held-out re-eval bank. Symlink the same realization files into the benchmark `data/fjsp/...` directory if you only generated one copy.

After a 600s run has **20 FEASIBLE** rows:

```bash
bash scripts/watch_cp600s_and_fold.sh
PYTHONPATH=src python scripts/fold_cp600s_into_compare.py
PYTHONPATH=src python scripts/rebuild_p0_baselines_compare.py
```

`watch_cp600s_and_fold.sh` attaches to a live `run_cp_stoch_p0.py --time-limit 600` process if one exists, otherwise resumes, then calls the fold script. Fold and rebuild share `sfjsp_alns.cp_fold.fold_cp600`. If the raw CSV is missing or has fewer than 20 FEASIBLE rows, they exit 2 and write nothing. ALNS and full-100 cells stay null unless `results/p0-sd3-10x5-summary.json`, `results/p0-sd3-10x5-raw.csv`, or `results/full-sd3-10x5-summary.json` are actually present. SPM-DAN stays null.

## Tests

```bash
PYTHONPATH=src pytest
```

`pyproject.toml` also sets `pythonpath = ["src"]`, so a plain `pytest` from the repository root is enough after `pip install -r requirements.txt`.

The smoke does not need the SD3 tree and does not assert paper makespans:

| Test | What it checks |
|---|---|
| `tests/test_fjs_io.py` | `.fjs` codec round-trip on `data/fixtures/synthetic/` |
| `tests/test_types_loader_smoke.py` | Both tiny fixtures load (2 jobs, 2 machines, 4 operations) |
| `tests/test_m0_decode.py` | MOR dispatch and decoder on `tiny_001.fjs`, scored on an in-process scenario pool |
| `tests/test_k_eval.py` | K_eval CLI: 2 synthetic instances, 3 ALNS iterations, disjoint slices, `comparable_to_paper_protocol` false, SPM-DAN stays N/A |
| `tests/test_cp_fold.py` | Shipped summaries keep SPM-DAN as N/A; the K_eval=1000 summary stays the 20/100 partial; `fold_cp600` writes nothing when the 600s CSV is missing or short |

Optional scripts that still need the NCO tree (and, for the PDR smoke, torch) exit with a pointer at the synthetic harness instead of a stack trace:

- `scripts/alns_m1_smoke.py`
- `scripts/sfjsp_smoke_eval.py`
