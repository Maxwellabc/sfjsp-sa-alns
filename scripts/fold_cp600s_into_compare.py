#!/usr/bin/env python3
"""Fold finished CP-600s + full-100 into compare artifacts and §5 docs.
Run only when p0-cp-stoch-600s-raw.csv has 20 FEASIBLE rows.
Does NOT rewrite Abstract/§1/§2 of paper-draft-sfjsp-alns.md.
"""
from __future__ import annotations
import csv, json, math, re
from pathlib import Path

ROOT = Path("/workspace/fjsp-combo-b")
RES = ROOT / "results"

def fnum(x):
    return float(x)

def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def main():
    raw = load_csv(RES / "p0-cp-stoch-600s-raw.csv")
    feas = [r for r in raw if r.get("status") == "FEASIBLE" and r.get("eval_mean_cmax")]
    if len(feas) < 20:
        raise SystemExit(f"need 20 FEASIBLE, got {len(feas)}")

    # Ensure summary/md exist (script usually writes them; regenerate defensively)
    means = [fnum(r["eval_mean_cmax"]) for r in feas]
    vars_ = [fnum(r["eval_var95"]) for r in feas]
    objs = [fnum(r["cp_obj_search"]) for r in feas]
    walls = [fnum(r["cp_wall_s"]) for r in feas if r.get("cp_wall_s")]
    macro_e = sum(means) / len(means)
    macro_v = sum(vars_) / len(vars_)
    macro_obj = sum(objs) / len(objs)
    mean_wall = sum(walls) / len(walls) if walls else None
    wall_total = sum(walls) if walls else None

    note = (
        "CP-stoch official code (FJSPmodel_stoch mean). "
        "Search on first K realizations; schedule decoded & re-eval "
        "on P0 held-out [250:350] with SA-ALNS decoder. "
        "Time limit 600s labeled **medium-budget** — still << paper 3600s."
    )
    summary = {
        "protocol": {
            "instances": 20,
            "time_limit_s": 600.0,
            "num_realizations_cp": 10,
            "workers": 2,
            "obj": "mean",
            "eval_slice": [250, 350],
            "budget_label": "medium",
            "out_prefix": "p0-cp-stoch-600s",
            "note": note,
        },
        "n_feasible": len(feas),
        "n_failed": len(raw) - len(feas),
        "macro_mean_cmax": macro_e,
        "macro_mean_var95": macro_v,
        "macro_cp_obj_search": macro_obj,
        "mean_wall_s": mean_wall,
        "wall_total_s": wall_total,
        "per_instance": [
            {
                "instance": r["instance"],
                "status": r["status"],
                "cp_obj_search": fnum(r["cp_obj_search"]),
                "cp_wall_s": fnum(r["cp_wall_s"]) if r.get("cp_wall_s") else None,
                "solution_count": int(float(r["solution_count"])) if r.get("solution_count") else 0,
                "eval_mean_cmax": fnum(r["eval_mean_cmax"]),
                "eval_var95": fnum(r["eval_var95"]),
                "num_realizations_cp": 10,
                "time_limit_s": 600.0,
                "workers": 2,
                "error": r.get("error", ""),
            }
            for r in feas
        ],
        "spmdan_status": (
            "N/A — blocked: no pretrained SPM-DAN/DAN-stoch checkpoints in "
            "NCO-for-Stochastic-FJSP (GitHub releases empty for FJSP weights); "
            "short CPU train unfair vs paper and missing deps "
            "(gym, torch_geometric, einops) at probe time."
        ),
    }
    (RES / "p0-cp-stoch-600s-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md_lines = [
        "# P0 CP-stoch (medium budget 600s) vs SA-ALNS protocol",
        "",
        "- Instances: 20 × SD3 `10x5+mix`",
        "- CP: OR-Tools CP-SAT stochastic **mean**, K_cp=10, time_limit=600s, workers=2",
        "- Eval: re-decode CP assignment+sequence on scenarios [250:350] (same decoder as P0)",
        f"- Feasible: {len(feas)}/20",
        f"- Macro E[Cmax] (eval): {macro_e:.4f}",
        f"- Macro VaR95 (eval): {macro_v:.4f}",
        f"- Macro CP search obj: {macro_obj:.4f}",
        f"- Mean solve wall: {mean_wall:.3f}s; total {wall_total:.1f}s" if mean_wall else "- Mean solve wall: n/a",
        "- Label: **medium-budget**; do not equate to paper 3600s",
        "",
        "| Instance | status | CP search obj | eval E[Cmax] | VaR95 | wall_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in feas:
        md_lines.append(
            f"| {r['instance']} | {r['status']} | {r['cp_obj_search']} | "
            f"{r['eval_mean_cmax']} | {r['eval_var95']} | {r['cp_wall_s']} |"
        )
    md_lines += ["", "## SPM-DAN", summary["spmdan_status"], ""]
    (RES / "p0-cp-stoch-600s.md").write_text("\n".join(md_lines) + "\n")

    # Load P0 ALNS / MOR / CP60 from existing compare CSV
    p0_cmp = load_csv(RES / "p0-baselines-compare.csv")
    keys = list(p0_cmp[0].keys())
    print("compare csv keys:", keys)
    # Canonical P0 columns (existing file uses *_E names)
    col_mor = "MOR_E" if "MOR_E" in keys else next(k for k in keys if "MOR" in k.upper())
    col_alns = "SA_ALNS_E" if "SA_ALNS_E" in keys else next(
        k for k in keys if "ALNS" in k.upper() and "minus" not in k.lower()
    )
    col_cp60 = "CP_stoch60_E" if "CP_stoch60_E" in keys else next(
        k for k in keys if "60" in k and "600" not in k
    )

    cp600_map = {r["instance"]: fnum(r["eval_mean_cmax"]) for r in feas}

    rows_out = []
    for r in p0_cmp:
        inst = r["instance"].replace(".fjs", "")
        mor = fnum(r[col_mor])
        alns = fnum(r[col_alns])
        c60 = fnum(r[col_cp60])
        c600 = cp600_map[inst]
        rows_out.append({
            "instance": inst,
            "MOR_E": mor,
            "SA_ALNS_E": alns,
            "CP_stoch60_E": c60,
            "CP_stoch600_E": c600,
            "ALNS_minus_CP60": alns - c60,
            "ALNS_minus_CP600": alns - c600,
            "CP600_minus_CP60": c600 - c60,
            "ALNS_minus_MOR": alns - mor,
        })

    macro_mor = sum(r["MOR_E"] for r in rows_out)/20
    macro_alns = sum(r["SA_ALNS_E"] for r in rows_out)/20
    macro_c60 = sum(r["CP_stoch60_E"] for r in rows_out)/20
    macro_c600 = sum(r["CP_stoch600_E"] for r in rows_out)/20

    # VaR from prior summary + new
    old = json.loads((RES / "p0-baselines-compare-summary.json").read_text())
    var_mor = old["macro_VaR95"]["MOR"]
    var_alns = old["macro_VaR95"]["SA-ALNS"]
    var_c60 = old["macro_VaR95"]["CP-stoch-60s-K10"]
    var_c600 = macro_v

    beats60 = sum(1 for r in rows_out if r["SA_ALNS_E"] < r["CP_stoch60_E"])
    beats600 = sum(1 for r in rows_out if r["SA_ALNS_E"] < r["CP_stoch600_E"])
    c600_beats_c60 = sum(1 for r in rows_out if r["CP_stoch600_E"] < r["CP_stoch60_E"])

    def pct(a, b):
        return 100.0 * (a - b) / b

    compare_summary = {
        "macro_E": {
            "MOR": macro_mor,
            "SA-ALNS": macro_alns,
            "CP-stoch-60s-K10": macro_c60,
            "CP-stoch-600s-K10": macro_c600,
            "SPM-DAN": None,
        },
        "macro_VaR95": {
            "MOR": var_mor,
            "SA-ALNS": var_alns,
            "CP-stoch-60s-K10": var_c60,
            "CP-stoch-600s-K10": var_c600,
            "SPM-DAN": None,
        },
        "vs_MOR_pct": {
            "SA-ALNS": pct(macro_alns, macro_mor),
            "CP-stoch-60s-K10": pct(macro_c60, macro_mor),
            "CP-stoch-600s-K10": pct(macro_c600, macro_mor),
        },
        "vs_ALNS_pct": {
            "CP-stoch-60s-K10": pct(macro_c60, macro_alns),
            "CP-stoch-600s-K10": pct(macro_c600, macro_alns),
        },
        "cp600_vs_cp60_pct": pct(macro_c600, macro_c60),
        "alns_beats_cp60_instances": beats60,
        "alns_beats_cp600_instances": beats600,
        "cp600_beats_cp60_instances": c600_beats_c60,
        "n_instances": 20,
        "cp60_protocol": old.get("cp_protocol"),
        "cp600_protocol": summary["protocol"],
        "spmdan": old.get("spmdan"),
        "full100_reminder": {
            "SA-ALNS_macro_E": 582.2680333333333,
            "MOR_macro_E": 641.8862333333334,
            "vs_MOR_pct": -9.29,
            "artifact": "results/full-sd3-10x5-summary.json",
            "note": "Separate bank/suite from P0-20; do not mix into P0 macro table",
        },
    }
    (RES / "p0-baselines-compare-summary.json").write_text(json.dumps(compare_summary, indent=2) + "\n")

    fieldnames = list(rows_out[0].keys())
    with (RES / "p0-baselines-compare.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    md = [
        "# P0 baselines comparison (MOR / SA-ALNS / CP-stoch-60s / CP-stoch-600s / SPM-DAN)",
        "",
        "Protocol: 20× SD3 `10x5+mix`, eval `[250:350]` K=100, same decoder.",
        "",
        "| Method | E[Cmax] | VaR95 | vs MOR (%) | notes |",
        "|---|---:|---:|---:|---|",
        f"| MOR | {macro_mor:.2f} | {var_mor:.2f} | +0.00 | PDR |",
        f"| **SA-ALNS** | **{macro_alns:.2f}** | **{var_alns:.2f}** | **{pct(macro_alns, macro_mor):.2f}** | 300 iter / 40s, K_search=50 |",
        f"| CP-stoch (short) | {macro_c60:.2f} | {var_c60:.2f} | {pct(macro_c60, macro_mor):.2f} | official mean CP-SAT, **60s**, K_cp=10; **not** paper 3600s |",
        f"| CP-stoch (medium) | {macro_c600:.2f} | {var_c600:.2f} | {pct(macro_c600, macro_mor):.2f} | official mean CP-SAT, **600s**, K_cp=10; **medium-budget**, still **not** paper 3600s |",
        "| SPM-DAN | N/A | N/A | — | blocked: no pretrained FJSP weights; short CPU train unfair / deps missing |",
        "",
        f"SA-ALNS beats CP-stoch-60s on **{beats60}/20** instances; beats CP-stoch-600s on **{beats600}/20**.",
        f"CP-600s improves on CP-60s on **{c600_beats_c60}/20** instances (macro Δ {pct(macro_c600, macro_c60):+.2f}% vs 60s).",
        "",
        "## Per-instance E[Cmax]",
        "",
        "| Instance | MOR | SA-ALNS | CP-60s | CP-600s | ALNS−CP60 | ALNS−CP600 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        md.append(
            f"| {r['instance']} | {r['MOR_E']:.2f} | {r['SA_ALNS_E']:.2f} | {r['CP_stoch60_E']:.2f} | "
            f"{r['CP_stoch600_E']:.2f} | {r['ALNS_minus_CP60']:.2f} | {r['ALNS_minus_CP600']:.2f} |"
        )
    md += [
        "",
        "**Caveat:** CP search objective on K=10 is optimistic vs held-out eval. "
        "Paper CP uses up to 3600s and often VaR; **60s = short**, **600s = medium** — neither is paper budget.",
        "",
        "## Full-100 reminder (separate suite; not mixed into P0 macros)",
        "",
        "- Artifact: `results/full-sd3-10x5-summary.json`",
        "- SA-ALNS macro E=**582.27**, MOR **641.89** (−9.29%); VaR95 ALNS **666.45** / MOR **749.98**",
        "- Realization bank note: official lognormal random-CoV, seed=200, NUM_SAMPLES=450 (same recipe family as P0).",
        "",
    ]
    (RES / "p0-baselines-compare.md").write_text("\n".join(md) + "\n")

    print(json.dumps({
        "macro_e_cp600": macro_e,
        "macro_v_cp600": macro_v,
        "macro_alns_p0": macro_alns,
        "macro_c60": macro_c60,
        "beats600": beats600,
        "beats60": beats60,
        "wall_total": wall_total,
        "mean_wall": mean_wall,
        "cp600_vs_alns_pct": pct(macro_c600, macro_alns),
        "cp600_vs_cp60_pct": pct(macro_c600, macro_c60),
    }, indent=2))

if __name__ == "__main__":
    main()
