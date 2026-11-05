#!/usr/bin/env python3
"""Rebuild results/p0-baselines-compare.* from MOR/ALNS + CP-60s + CP-600s."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/workspace/fjsp-combo-b")
RES = ROOT / "results"


def load_alns_per_inst():
    raw = RES / "p0-sd3-10x5-raw.csv"
    # aggregate seed-mean E and VaR for MOR and SA-ALNS
    by = {}
    with raw.open() as f:
        for row in csv.DictReader(f):
            inst = row["instance"].replace(".fjs", "")
            method = row["method"]
            if method not in ("MOR", "SA-ALNS"):
                continue
            by.setdefault(inst, {}).setdefault(method, []).append(
                (float(row["mean_cmax"]), float(row["var95"]))
            )
    out = {}
    for inst, methods in by.items():
        out[inst] = {}
        for m, vals in methods.items():
            em = sum(v[0] for v in vals) / len(vals)
            ev = sum(v[1] for v in vals) / len(vals)
            out[inst][m] = (em, ev)
    return out


def load_cp(prefix: str):
    path = RES / f"{prefix}-summary.json"
    data = json.loads(path.read_text())
    per = {}
    for r in data["per_instance"]:
        per[r["instance"]] = (
            r.get("eval_mean_cmax"),
            r.get("eval_var95"),
            r.get("status"),
        )
    return data, per


def main():
    alns = load_alns_per_inst()
    alns_sum = json.loads((RES / "p0-sd3-10x5-summary.json").read_text())
    cp60, cp60_per = load_cp("p0-cp-stoch")
    cp600, cp600_per = load_cp("p0-cp-stoch-600s")

    insts = sorted(alns.keys())
    mor_E = alns_sum["macro_mean_cmax"]["MOR"]
    mor_V = alns_sum["macro_mean_var95"]["MOR"]
    alns_E = alns_sum["macro_mean_cmax"]["SA-ALNS"]
    alns_V = alns_sum["macro_mean_var95"]["SA-ALNS"]
    cp60_E = cp60["macro_mean_cmax"]
    cp60_V = cp60["macro_mean_var95"]
    cp600_E = cp600["macro_mean_cmax"]
    cp600_V = cp600["macro_mean_var95"]

    def vs_mor(x):
        return 100.0 * (x - mor_E) / mor_E

    wins_60 = 0
    wins_600 = 0
    rows = []
    for inst in insts:
        mor_e, _ = alns[inst]["MOR"]
        alns_e, _ = alns[inst]["SA-ALNS"]
        c60 = cp60_per[inst][0]
        c600 = cp600_per[inst][0]
        d60 = alns_e - c60 if c60 is not None else None
        d600 = alns_e - c600 if c600 is not None else None
        if d60 is not None and d60 < 0:
            wins_60 += 1
        if d600 is not None and d600 < 0:
            wins_600 += 1
        rows.append(
            {
                "instance": inst,
                "MOR": round(mor_e, 2),
                "SA-ALNS": round(alns_e, 2),
                "CP-stoch-60s": round(c60, 2) if c60 is not None else None,
                "CP-stoch-600s": round(c600, 2) if c600 is not None else None,
                "ALNS-CP60": round(d60, 2) if d60 is not None else None,
                "ALNS-CP600": round(d600, 2) if d600 is not None else None,
            }
        )

    summary = {
        "macro_E": {
            "MOR": mor_E,
            "SA-ALNS": alns_E,
            "CP-stoch-60s-K10": cp60_E,
            "CP-stoch-600s-K10": cp600_E,
            "SPM-DAN": None,
        },
        "macro_VaR95": {
            "MOR": mor_V,
            "SA-ALNS": alns_V,
            "CP-stoch-60s-K10": cp60_V,
            "CP-stoch-600s-K10": cp600_V,
            "SPM-DAN": None,
        },
        "vs_MOR_pct": {
            "SA-ALNS": vs_mor(alns_E),
            "CP-stoch-60s-K10": vs_mor(cp60_E),
            "CP-stoch-600s-K10": vs_mor(cp600_E),
        },
        "alns_vs_cp600_pct": 100.0 * (alns_E - cp600_E) / cp600_E,
        "alns_beats_cp60_instances": wins_60,
        "alns_beats_cp600_instances": wins_600,
        "n_instances": len(insts),
        "cp60_protocol": cp60["protocol"],
        "cp600_protocol": cp600["protocol"],
        "spmdan": cp600.get("spmdan_status") or cp60.get("spmdan_status"),
        "honesty": (
            "CP-60s = short budget; CP-600s = medium budget (10× short). "
            "Neither is paper 3600s. SPM-DAN = N/A."
        ),
    }
    (RES / "p0-baselines-compare-summary.json").write_text(json.dumps(summary, indent=2))

    with (RES / "p0-baselines-compare.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    alns_vs_600 = 100.0 * (alns_E - cp600_E) / cp600_E
    md = f"""# P0 baselines comparison (MOR / SA-ALNS / CP-stoch-60s / CP-stoch-600s / SPM-DAN)

Protocol: 20× SD3 `10x5+mix`, eval `[250:350]` K=100, same decoder.

| Method | E[Cmax] | VaR95 | vs MOR (%) | notes |
|---|---:|---:|---:|---|
| MOR | {mor_E:.2f} | {mor_V:.2f} | +0.00 | PDR |
| **SA-ALNS** | **{alns_E:.2f}** | **{alns_V:.2f}** | **{vs_mor(alns_E):.2f}** | 300 iter / ~40s, K_search=50 |
| CP-stoch (short 60s) | {cp60_E:.2f} | {cp60_V:.2f} | {vs_mor(cp60_E):.2f} | official mean CP-SAT, **60s**, K_cp=10; **not** paper 3600s |
| CP-stoch (medium 600s) | {cp600_E:.2f} | {cp600_V:.2f} | {vs_mor(cp600_E):.2f} | same protocol, **600s** (10× short); still **<<** paper 3600s |
| SPM-DAN | N/A | N/A | — | blocked: no pretrained FJSP weights; short CPU train unfair / deps missing |

SA-ALNS vs CP-stoch-60s: wins **{wins_60}/{len(insts)}** instances (seed-mean E vs single CP).
SA-ALNS vs CP-stoch-600s: wins **{wins_600}/{len(insts)}** instances; macro Δ = **{alns_vs_600:.2f}%** (negative ⇒ ALNS better).

## Per-instance E[Cmax]

| Instance | MOR | SA-ALNS | CP-60s | CP-600s | ALNS−CP60 | ALNS−CP600 |
|---|---:|---:|---:|---:|---:|---:|
"""
    for r in rows:
        md += (
            f"| {r['instance']} | {r['MOR']:.2f} | {r['SA-ALNS']:.2f} | "
            f"{r['CP-stoch-60s']:.2f} | {r['CP-stoch-600s']:.2f} | "
            f"{r['ALNS-CP60']:.2f} | {r['ALNS-CP600']:.2f} |\n"
        )
    md += """
**Caveats:** CP search uses K_cp=10 (optimistic vs held-out). 60s = short; 600s = medium. Paper CP uses up to 3600s and often VaR — **do not** claim win over Smit 3600s CP. SPM-DAN remains N/A.
"""
    (RES / "p0-baselines-compare.md").write_text(md)
    print(json.dumps({k: summary[k] for k in summary if "protocol" not in k}, indent=2))


if __name__ == "__main__":
    main()
