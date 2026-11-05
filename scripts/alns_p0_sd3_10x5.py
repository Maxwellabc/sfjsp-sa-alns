#!/usr/bin/env python3
"""P0: SD3 10x5 subset — PDRs vs SA-ALNS (D1+D3–D5), multi-seed, K_eval=100/200."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/fjsp-combo-b")
sys.path.insert(0, str(ROOT / "src"))

from sfjsp_alns.alns import ALNSConfig, run_alns
from sfjsp_alns.evaluate import evaluate_solution
from sfjsp_alns.init_pdr import best_pdr_seed, pdr_solution
from sfjsp_alns.io.fjsp_loader import load_instance
from sfjsp_alns.types import ScenarioBundle

INST_DIR = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/SD3/10x5+mix"
REAL_DIR = (
    ROOT
    / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/stochastic_realizations/random/SD3/10x5+mix"
)
OUT_DIR = ROOT / "results"
OUT_CSV = OUT_DIR / "p0-sd3-10x5-raw.csv"
OUT_MD = OUT_DIR / "p0-sd3-10x5.md"
OUT_JSON = OUT_DIR / "p0-sd3-10x5-summary.json"

SEEDS = [50, 51, 52]
EVAL_START = 250
EVAL_K = 100  # [250:350]; generated pool has 450
SEARCH_START, SEARCH_K = 0, 50
MAX_INST = 20
ALNS_ITERS = 300
ALNS_TIME = 40.0
DESTROY = ("random_ops", "critical_path", "high_var_ops", "machine_load")
REPAIR = ("spt_biased",)
PDRS = ("FIFO", "MOR", "SPT", "MWKR")


def load_pool(path: Path) -> np.ndarray:
    drl = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL"
    sys.path.insert(0, str(drl))
    from data_utils import load_data_from_stochastic_realizations_file

    _, op_pts = load_data_from_stochastic_realizations_file(str(path), max_realizations=500)
    return np.stack([np.asarray(p, dtype=np.int64) for p in op_pts], axis=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(REAL_DIR.glob("*.fjs"))[:MAX_INST]
    if not names:
        raise SystemExit(f"no realizations in {REAL_DIR}")

    rows: list[dict] = []
    t0 = time.time()
    print(f"P0: {len(names)} inst × {len(SEEDS)} seeds; K_eval={EVAL_K}; destroy={DESTROY}")

    for path in names:
        inst = load_instance(INST_DIR / path.name)
        pool = load_pool(path)
        if pool.shape[0] < EVAL_START + EVAL_K:
            raise RuntimeError(f"{path.name}: only {pool.shape[0]} realizations")
        eval_sc = ScenarioBundle(
            op_pt_scen=pool[EVAL_START : EVAL_START + EVAL_K],
            role="eval",
            source_slice=(EVAL_START, EVAL_START + EVAL_K),
        )
        search_sc = ScenarioBundle(
            op_pt_scen=pool[SEARCH_START : SEARCH_START + SEARCH_K],
            role="search",
            source_slice=(SEARCH_START, SEARCH_START + SEARCH_K),
        )

        # PDRs are deterministic given seed+nominal; evaluate once per seed for fairness
        for seed in SEEDS:
            for pdr in PDRS:
                sol = pdr_solution(inst, pdr, seed=seed)
                m = evaluate_solution(inst, sol, eval_sc)
                rows.append(
                    {
                        "instance": path.name,
                        "method": pdr,
                        "seed": seed,
                        "mean_cmax": m.mean_cmax,
                        "var95": m.var_cmax,
                        "iters": 0,
                        "seconds": 0.0,
                    }
                )

            init = best_pdr_seed(inst, search_sc, seed=seed)
            cfg = ALNSConfig(
                max_iter=ALNS_ITERS,
                time_limit_s=ALNS_TIME,
                seed=seed + (hash(path.name) % 997),
                k_fast=10,
                local_machine_tries=20,
                destroy_names=DESTROY,
                repair_names=REPAIR,
            )
            t1 = time.time()
            res = run_alns(inst, init, search_sc, scenarios_refine=search_sc, config=cfg)
            elapsed = time.time() - t1
            m = evaluate_solution(inst, res.best, eval_sc)
            rows.append(
                {
                    "instance": path.name,
                    "method": "SA-ALNS",
                    "seed": seed,
                    "mean_cmax": m.mean_cmax,
                    "var95": m.var_cmax,
                    "iters": res.iters,
                    "seconds": elapsed,
                }
            )
            print(
                f"  {path.name} seed={seed}: ALNS E={m.mean_cmax:.2f} VaR={m.var_cmax:.2f} "
                f"iters={res.iters} {elapsed:.1f}s"
            )

    # write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["instance", "method", "seed", "mean_cmax", "var95", "iters", "seconds"],
        )
        w.writeheader()
        w.writerows(rows)

    # aggregate: mean over seeds per (instance, method), then mean over instances
    methods = list(PDRS) + ["SA-ALNS"]
    by_method: dict[str, list[float]] = {m: [] for m in methods}
    by_method_var: dict[str, list[float]] = {m: [] for m in methods}
    per_inst: dict[str, dict[str, dict[str, float]]] = {}

    for inst_name in sorted({r["instance"] for r in rows}):
        per_inst[inst_name] = {}
        for method in methods:
            sub = [r for r in rows if r["instance"] == inst_name and r["method"] == method]
            mean_e = float(np.mean([r["mean_cmax"] for r in sub]))
            std_e = float(np.std([r["mean_cmax"] for r in sub], ddof=0))
            mean_v = float(np.mean([r["var95"] for r in sub]))
            per_inst[inst_name][method] = {
                "mean_cmax": mean_e,
                "std_cmax": std_e,
                "var95": mean_v,
                "seconds": float(np.mean([r["seconds"] for r in sub])),
            }
            by_method[method].append(mean_e)
            by_method_var[method].append(mean_v)

    summary = {
        "n_instances": len(per_inst),
        "seeds": SEEDS,
        "k_eval": EVAL_K,
        "eval_slice": [EVAL_START, EVAL_START + EVAL_K],
        "search_slice": [SEARCH_START, SEARCH_START + SEARCH_K],
        "destroy": list(DESTROY),
        "repair": list(REPAIR),
        "alns_max_iter": ALNS_ITERS,
        "wall_total_s": time.time() - t0,
        "macro_mean_cmax": {m: float(np.mean(by_method[m])) for m in methods},
        "macro_mean_var95": {m: float(np.mean(by_method_var[m])) for m in methods},
        "spmdan": "N/A — pretrained weights unavailable",
        "cp_stoch": "N/A — not run in this P0 pass",
        "realization_note": (
            f"Generated with official lognormal random-CoV recipe, seed={200}, "
            f"NUM_SAMPLES=450 under {REAL_DIR} (not identical to smoke_SD3 bank)."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown report
    lines: list[str] = []
    def L(s: str = "") -> None:
        lines.append(s)
    L("# P0 Experiment — SD3 10×5 subset (SA-ALNS vs PDRs)")
    L()
    L(f"- Time: 2026-11-05 (Asia/Shanghai)")
    L(f"- Instances: **{len(per_inst)}** (`10x5+mix_001` … `_{len(per_inst):03d}`)")
    L(f"- Seeds: {SEEDS} (report mean±std of E[Cmax] across seeds per instance, then macro-average)")
    L(f"- Eval: realizations `[{EVAL_START}:{EVAL_START+EVAL_K}]` (K_eval={EVAL_K})")
    L(f"- Search: `[{SEARCH_START}:{SEARCH_START+SEARCH_K}]` (no overlap with eval)")
    L(f"- SA-ALNS: destroy={list(DESTROY)}, repair={list(REPAIR)}, max_iter={ALNS_ITERS}, tlim={ALNS_TIME}s")
    L(f"- SPM-DAN / CP-stoch: **N/A** (weights missing / not run)")
    L(f"- Realization bank: newly generated (`NUM_SAMPLES=450`, seed_datagen=200); differs from `smoke_SD3`")
    L(f"- Wall clock: {summary['wall_total_s']:.1f}s")
    L()
    L("## Macro averages (mean over instances of per-instance seed-means)")
    L()
    L("| Method | E[Cmax] | VaR95 | vs MOR gap% |")
    L("|---|---:|---:|---:|")
    mor = summary["macro_mean_cmax"]["MOR"]
    for m in methods:
        e = summary["macro_mean_cmax"][m]
        v = summary["macro_mean_var95"][m]
        gap = (e - mor) / mor * 100
        L(f"| {m} | {e:.2f} | {v:.2f} | {gap:+.2f} |")
    L()
    L("## Per-instance seed-mean E[Cmax]")
    L()
    header = "| Instance | " + " | ".join(methods) + " | ALNS−MOR |"
    L(header)
    L("|---|" + "|".join(["---:"] * (len(methods) + 1)) + "|")
    for inst_name, md in per_inst.items():
        cells = [f"{md[m]['mean_cmax']:.2f}" for m in methods]
        delta = md["SA-ALNS"]["mean_cmax"] - md["MOR"]["mean_cmax"]
        L(f"| {inst_name} | " + " | ".join(cells) + f" | {delta:+.2f} |")
    alns_beats = sum(
        1 for md in per_inst.values() if md["SA-ALNS"]["mean_cmax"] < md["MOR"]["mean_cmax"]
    )
    L()
    L("## Notes")
    L()
    L(f"- SA-ALNS beats package MOR on **{alns_beats}/{len(per_inst)}** instances (seed-mean E[Cmax]).")
    L(f"- Raw CSV: `{OUT_CSV}`")
    L(f"- Summary JSON: `{OUT_JSON}`")
    L("- Paper K_eval=1000 and full 100-instance suite remain outstanding.")
    L()
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_MD}")
    print("Macro:", summary["macro_mean_cmax"])


if __name__ == "__main__":
    main()
