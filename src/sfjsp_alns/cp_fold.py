"""Fold a finished CP-stoch 600s run into compare artifacts.

Numbers are read from files under the results directory. Missing ALNS, CP-60s,
or full-100 JSON is left null. This module never fills those cells from the
experiments draft.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SPMDAN_STATUS = (
    "N/A — blocked: no pretrained SPM-DAN/DAN-stoch checkpoints in "
    "NCO-for-Stochastic-FJSP. Do not impute a score."
)

MIN_FEASIBLE = 20


class FoldError(RuntimeError):
    """Raised when the 600s artifact is absent or incomplete."""


def _fnum(value: object) -> float:
    if value is None or value == "":
        raise ValueError("empty numeric field")
    return float(value)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    if not values:
        raise FoldError("cannot average an empty column")
    return sum(values) / len(values)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _cp600_from_raw(raw_path: Path) -> tuple[dict, list[dict]]:
    rows = _load_csv(raw_path)
    feas = [
        r
        for r in rows
        if r.get("status") == "FEASIBLE" and str(r.get("eval_mean_cmax") or "").strip()
    ]
    if len(feas) < MIN_FEASIBLE:
        raise FoldError(
            f"{raw_path.name} has {len(feas)} FEASIBLE rows with eval_mean_cmax; "
            f"need {MIN_FEASIBLE}. Resume scripts/run_cp_stoch_p0.py "
            "--time-limit 600 --out-prefix p0-cp-stoch-600s --resume. "
            "Refusing to impute the missing instances."
        )
    means = [_fnum(r["eval_mean_cmax"]) for r in feas]
    vars_ = [_fnum(r["eval_var95"]) for r in feas]
    objs = [_fnum(r["cp_obj_search"]) for r in feas]
    walls = [_fnum(r["cp_wall_s"]) for r in feas if str(r.get("cp_wall_s") or "").strip()]
    per = []
    for r in feas:
        per.append(
            {
                "instance": r["instance"].replace(".fjs", ""),
                "status": r["status"],
                "cp_obj_search": _fnum(r["cp_obj_search"]),
                "cp_wall_s": _fnum(r["cp_wall_s"]) if str(r.get("cp_wall_s") or "").strip() else None,
                "solution_count": int(float(r["solution_count"]))
                if str(r.get("solution_count") or "").strip()
                else 0,
                "eval_mean_cmax": _fnum(r["eval_mean_cmax"]),
                "eval_var95": _fnum(r["eval_var95"]),
                "num_realizations_cp": int(float(r["num_realizations_cp"]))
                if str(r.get("num_realizations_cp") or "").strip()
                else None,
                "time_limit_s": _fnum(r["time_limit_s"])
                if str(r.get("time_limit_s") or "").strip()
                else 600.0,
                "error": r.get("error", ""),
            }
        )
    note = (
        "CP-stoch official code (FJSPmodel_stoch mean). "
        "Search on the first K realizations; schedule re-decoded on the held-out "
        "eval slice. Time limit is medium-budget 600s, not the paper 3600s. "
        "Macros are the mean of FEASIBLE rows in the raw CSV only."
    )
    summary = {
        "protocol": {
            "instances": len(per),
            "time_limit_s": per[0]["time_limit_s"],
            "num_realizations_cp": per[0]["num_realizations_cp"],
            "obj": "mean",
            "budget_label": "medium",
            "out_prefix": "p0-cp-stoch-600s",
            "source_csv": raw_path.name,
            "note": note,
        },
        "n_feasible": len(per),
        "n_failed": len(rows) - len(feas),
        "macro_mean_cmax": _mean(means),
        "macro_mean_var95": _mean(vars_),
        "macro_cp_obj_search": _mean(objs),
        "mean_wall_s": _mean(walls) if walls else None,
        "wall_total_s": sum(walls) if walls else None,
        "per_instance": per,
        "spmdan_status": SPMDAN_STATUS,
    }
    return summary, per


def _maybe_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _alns_from_raw(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """instance -> method -> (mean E, mean VaR) over seeds."""
    if not path.is_file():
        return {}
    bucket: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in _load_csv(path):
        method = row.get("method")
        if method not in ("MOR", "SA-ALNS"):
            continue
        inst = row["instance"].replace(".fjs", "")
        bucket.setdefault(inst, {}).setdefault(method, []).append(
            (_fnum(row["mean_cmax"]), _fnum(row["var95"]))
        )
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for inst, methods in bucket.items():
        out[inst] = {}
        for method, vals in methods.items():
            out[inst][method] = (
                _mean([v[0] for v in vals]),
                _mean([v[1] for v in vals]),
            )
    return out


def _pct(new: float, base: float) -> float:
    return 100.0 * (new - base) / base


def fold_cp600(results_dir: Path) -> dict:
    """Write 600s summary/markdown and a compare file. Return the compare dict.

    Requires ``p0-cp-stoch-600s-raw.csv`` with at least 20 FEASIBLE rows.
    """
    results_dir = Path(results_dir)
    raw_path = results_dir / "p0-cp-stoch-600s-raw.csv"
    if not raw_path.is_file():
        raise FoldError(
            f"missing {raw_path}. Run CP-stoch with --time-limit 600 "
            "--out-prefix p0-cp-stoch-600s. No CP-600s number is invented."
        )
    summary, per = _cp600_from_raw(raw_path)
    _write(
        results_dir / "p0-cp-stoch-600s-summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    _write_cp_md(results_dir / "p0-cp-stoch-600s.md", summary)

    cp60 = _maybe_json(results_dir / "p0-cp-stoch-summary.json")
    alns_sum = _maybe_json(results_dir / "p0-sd3-10x5-summary.json")
    full = _maybe_json(results_dir / "full-sd3-10x5-summary.json")
    alns_per = _alns_from_raw(results_dir / "p0-sd3-10x5-raw.csv")

    cp60_per: dict[str, tuple[float, float]] = {}
    if cp60 and isinstance(cp60.get("per_instance"), list):
        for row in cp60["per_instance"]:
            if row.get("eval_mean_cmax") is None:
                continue
            cp60_per[str(row["instance"]).replace(".fjs", "")] = (
                float(row["eval_mean_cmax"]),
                float(row["eval_var95"]),
            )

    cp600_per = {
        row["instance"]: (row["eval_mean_cmax"], row["eval_var95"]) for row in per
    }
    instances = sorted(set(cp600_per) | set(cp60_per) | set(alns_per))

    table = []
    for inst in instances:
        mor = alns_per.get(inst, {}).get("MOR")
        alns = alns_per.get(inst, {}).get("SA-ALNS")
        c60 = cp60_per.get(inst)
        c600 = cp600_per.get(inst)
        table.append(
            {
                "instance": inst,
                "MOR_E": None if mor is None else mor[0],
                "SA_ALNS_E": None if alns is None else alns[0],
                "CP60_E": None if c60 is None else c60[0],
                "CP600_E": None if c600 is None else c600[0],
                "ALNS_minus_MOR": None if mor is None or alns is None else alns[0] - mor[0],
                "ALNS_minus_CP60": None if alns is None or c60 is None else alns[0] - c60[0],
                "ALNS_minus_CP600": None
                if alns is None or c600 is None
                else alns[0] - c600[0],
                "CP600_minus_CP60": None if c60 is None or c600 is None else c600[0] - c60[0],
            }
        )

    def macro_from(key: str) -> float | None:
        vals = [row[key] for row in table if row[key] is not None]
        if not vals:
            return None
        return _mean([float(v) for v in vals])

    mor_e = None
    alns_e = None
    mor_v = None
    alns_v = None
    if alns_sum and isinstance(alns_sum.get("macro_mean_cmax"), dict):
        mor_e = alns_sum["macro_mean_cmax"].get("MOR")
        alns_e = alns_sum["macro_mean_cmax"].get("SA-ALNS")
        var = alns_sum.get("macro_mean_var95") or {}
        mor_v = var.get("MOR")
        alns_v = var.get("SA-ALNS")
    else:
        mor_e = macro_from("MOR_E")
        alns_e = macro_from("SA_ALNS_E")

    cp60_e = cp60.get("macro_mean_cmax") if cp60 else None
    cp60_v = cp60.get("macro_mean_var95") if cp60 else None
    cp600_e = summary["macro_mean_cmax"]
    cp600_v = summary["macro_mean_var95"]

    def beats(left: str, right: str) -> int | None:
        pairs = [
            (row[left], row[right])
            for row in table
            if row[left] is not None and row[right] is not None
        ]
        if not pairs:
            return None
        return sum(1 for a, b in pairs if a < b)

    compare = {
        "macro_E": {
            "MOR": mor_e,
            "SA-ALNS": alns_e,
            "CP-stoch-60s-K10": cp60_e,
            "CP-stoch-600s-K10": cp600_e,
            "SPM-DAN": None,
        },
        "macro_VaR95": {
            "MOR": mor_v,
            "SA-ALNS": alns_v,
            "CP-stoch-60s-K10": cp60_v,
            "CP-stoch-600s-K10": cp600_v,
            "SPM-DAN": None,
        },
        "vs_MOR_pct": {},
        "alns_beats_cp60_instances": beats("SA_ALNS_E", "CP60_E"),
        "alns_beats_cp600_instances": beats("SA_ALNS_E", "CP600_E"),
        "cp600_beats_cp60_instances": beats("CP600_E", "CP60_E"),
        "n_instances_cp600": summary["n_feasible"],
        "sources": {
            "cp600": raw_path.name,
            "cp60": "p0-cp-stoch-summary.json" if cp60 else None,
            "alns_p0_summary": "p0-sd3-10x5-summary.json" if alns_sum else None,
            "alns_p0_raw": "p0-sd3-10x5-raw.csv" if alns_per else None,
            "full100": "full-sd3-10x5-summary.json" if full else None,
        },
        "cp600_protocol": summary["protocol"],
        "cp60_protocol": None if cp60 is None else cp60.get("protocol"),
        "spmdan": SPMDAN_STATUS,
        "honesty": (
            "CP-60s is a short budget and CP-600s is a medium budget. "
            "Neither is the paper 3600s run. Cells are null when the source "
            "JSON/CSV is not in this results directory."
        ),
    }
    if isinstance(mor_e, (int, float)) and mor_e:
        for label, value in (
            ("SA-ALNS", alns_e),
            ("CP-stoch-60s-K10", cp60_e),
            ("CP-stoch-600s-K10", cp600_e),
        ):
            if isinstance(value, (int, float)):
                compare["vs_MOR_pct"][label] = _pct(float(value), float(mor_e))
    if full and isinstance(full.get("macro_mean_cmax"), dict):
        compare["full100"] = {
            "artifact": "full-sd3-10x5-summary.json",
            "n_instances": full.get("n_instances"),
            "k_eval": full.get("k_eval"),
            "macro_mean_cmax": full.get("macro_mean_cmax"),
            "macro_mean_var95": full.get("macro_mean_var95"),
            "note": "Copied from full-sd3-10x5-summary.json. Not mixed into the P0 macro.",
        }
    else:
        compare["full100"] = None

    _write(
        results_dir / "p0-baselines-compare-summary.json",
        json.dumps(compare, indent=2) + "\n",
    )
    _write_compare_csv(results_dir / "p0-baselines-compare.csv", table)
    _write_compare_md(results_dir / "p0-baselines-compare.md", compare, table)
    return compare


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_compare_csv(path: Path, table: list[dict]) -> None:
    if not table:
        return
    fields = list(table[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({k: _fmt(row[k]) for k in fields})


def _md_cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_cp_md(path: Path, summary: dict) -> None:
    lines = [
        "# P0 CP-stoch (medium budget 600s)",
        "",
        "Aggregated only from FEASIBLE rows in `p0-cp-stoch-600s-raw.csv`.",
        "This is a medium budget, not the paper 3600s limit.",
        "",
        f"- Feasible: {summary['n_feasible']}",
        f"- Macro E[Cmax]: {summary['macro_mean_cmax']:.4f}",
        f"- Macro VaR95: {summary['macro_mean_var95']:.4f}",
        f"- Macro CP search obj: {summary['macro_cp_obj_search']:.4f}",
        f"- Mean wall: {summary['mean_wall_s']}",
        "",
        "| Instance | status | CP search obj | eval E[Cmax] | VaR95 | wall_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_instance"]:
        lines.append(
            f"| {row['instance']} | {row['status']} | {row['cp_obj_search']} | "
            f"{row['eval_mean_cmax']} | {row['eval_var95']} | {row['cp_wall_s']} |"
        )
    lines += ["", "## SPM-DAN", summary["spmdan_status"], ""]
    _write(path, "\n".join(lines) + "\n")


def _write_compare_md(path: Path, compare: dict, table: list[dict]) -> None:
    macro = compare["macro_E"]
    var = compare["macro_VaR95"]
    lines = [
        "# P0 baselines comparison",
        "",
        "Null cells are missing artifacts, not zeros. SPM-DAN is N/A.",
        "CP-60s = short budget. CP-600s = medium budget. Neither is paper 3600s.",
        "",
        "| Method | E[Cmax] | VaR95 | source |",
        "|---|---:|---:|---|",
        f"| MOR | {_md_cell(macro['MOR'])} | {_md_cell(var['MOR'])} | p0-sd3 summary/raw if present |",
        f"| SA-ALNS | {_md_cell(macro['SA-ALNS'])} | {_md_cell(var['SA-ALNS'])} | p0-sd3 summary/raw if present |",
        f"| CP-stoch 60s | {_md_cell(macro['CP-stoch-60s-K10'])} | {_md_cell(var['CP-stoch-60s-K10'])} | p0-cp-stoch-summary.json |",
        f"| CP-stoch 600s | {_md_cell(macro['CP-stoch-600s-K10'])} | {_md_cell(var['CP-stoch-600s-K10'])} | p0-cp-stoch-600s-raw.csv |",
        "| SPM-DAN | N/A | N/A | weights unavailable |",
        "",
        "## Per instance",
        "",
        "| Instance | MOR | SA-ALNS | CP-60s | CP-600s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['instance']} | {_md_cell(row['MOR_E'])} | {_md_cell(row['SA_ALNS_E'])} | "
            f"{_md_cell(row['CP60_E'])} | {_md_cell(row['CP600_E'])} |"
        )
    if compare.get("full100"):
        lines += [
            "",
            "## Full-100 reminder",
            "",
            "Copied from `full-sd3-10x5-summary.json`. Not mixed into the P0 macro.",
            "",
            "```json",
            json.dumps(compare["full100"], indent=2),
            "```",
        ]
    else:
        lines += [
            "",
            "## Full-100 reminder",
            "",
            "`results/full-sd3-10x5-summary.json` is not in this directory, so no full-100 macro is reported.",
            "",
        ]
    lines.append(compare["spmdan"])
    lines.append("")
    _write(path, "\n".join(lines) + "\n")
