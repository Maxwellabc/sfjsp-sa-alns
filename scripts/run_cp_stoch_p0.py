#!/usr/bin/env python3
"""P0-aligned CP-stoch (mean) vs SA-ALNS protocol.

Solve CP-SAT stochastic mean model on search realizations [0:K_search_cp],
convert assignment+start-order to our Solution encoding, evaluate on [250:350]
with the same decoder as P0 PDRs/ALNS.

Honest short/medium time limits (default 60s); never claim paper 3600s.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

ROOT = Path("/workspace/fjsp-combo-b")
CP_ROOT = (
    ROOT
    / "baselines/NCO-for-Stochastic-FJSP/Job_Shop_Scheduling_Benchmark_Environments_and_Instances"
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CP_ROOT))

from sfjsp_alns.evaluate import evaluate_solution
from sfjsp_alns.io.fjsp_loader import load_instance
from sfjsp_alns.types import ScenarioBundle, Solution
from solution_methods.cp_sat import FJSPmodel_stoch
from solution_methods.cp_sat import utils as cp_utils
from solution_methods.helper_functions import load_stochastic_job_shop_env

INST_DIR = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/SD3/10x5+mix"
REAL_DIR = (
    ROOT
    / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/stochastic_realizations/random/SD3/10x5+mix"
)
OUT_DIR = ROOT / "results"
EVAL_START, EVAL_K = 250, 100
INSTANCE_REL = "/fjsp/stoch/random/SD3/10x5+mix"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P0 CP-stoch (mean) baseline")
    p.add_argument(
        "--time-limit",
        type=float,
        default=60,
        help="CP-SAT wall seconds per instance (default 60 short-budget)",
    )
    p.add_argument(
        "--num-realizations",
        type=int,
        default=10,
        help="K_cp: first N realizations for CP search (default 10)",
    )
    p.add_argument(
        "--max-inst",
        type=int,
        default=20,
        help="Max instances (default 20 = 001..020)",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="p0-cp-stoch",
        help="Output stem under results/ (default p0-cp-stoch)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="OR-Tools num_search_workers (default 2; lower if RAM tight)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already FEASIBLE in existing raw CSV for out-prefix",
    )
    return p.parse_args()


def load_pool(path: Path) -> np.ndarray:
    """Load realization bank. FJSP-DRL modules parse sys.argv on import — sanitize."""
    drl = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL"
    sys.path.insert(0, str(drl))
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]]
        from data_utils import load_data_from_stochastic_realizations_file

        _, op_pts = load_data_from_stochastic_realizations_file(
            str(path), max_realizations=500
        )
    finally:
        sys.argv = saved_argv
    return np.stack([np.asarray(p, dtype=np.int64) for p in op_pts], axis=0)


def schedule_to_solution(instance, schedule) -> Solution:
    """Convert CP Schedule (job->tasks with start/machine) to Solution."""
    n_ops = instance.n_ops
    job_length = instance.job_length
    first = np.cumsum(np.concatenate([[0], job_length[:-1]])).astype(np.int64)
    machine_assign = np.full(n_ops, -1, dtype=np.int64)
    starts = np.full(n_ops, 0, dtype=np.int64)
    for job_info in schedule:
        j = int(job_info["job"])
        for task in job_info["tasks"]:
            t = int(task["task"])
            op = int(first[j] + t)
            machine_assign[op] = int(task["machine"])
            starts[op] = int(task["start"])
    if np.any(machine_assign < 0):
        raise RuntimeError("incomplete CP schedule")
    op_sequence = np.argsort(starts, kind="stable").astype(np.int64)
    job_of = np.repeat(np.arange(instance.n_jobs), job_length)
    next_needed = first.copy()
    ordered = []
    remaining = list(op_sequence.tolist())
    guard = 0
    while remaining and guard < n_ops * n_ops + 5:
        guard += 1
        progressed = False
        for idx, op in enumerate(remaining):
            j = int(job_of[op])
            if next_needed[j] == op:
                ordered.append(op)
                next_needed[j] = op + 1 if op + 1 <= first[j] + job_length[j] - 1 else op + 1
                remaining.pop(idx)
                progressed = True
                break
        if not progressed:
            ordered.extend(remaining)
            break
    return Solution(
        machine_assign=machine_assign,
        op_sequence=np.asarray(ordered, dtype=np.int64),
        meta={"source": "cp_stoch"},
    )


def solve_model_workers(model, time_limit: float, workers: int):
    """Like cp_utils.solve_model but with configurable search workers."""
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if workers and workers > 0:
        solver.parameters.num_search_workers = int(workers)
    solution_printer = cp_utils.SolutionPrinter()
    status = solver.Solve(model, solution_printer)
    solution_count = solution_printer.solution_count()
    return solver, status, solution_count


def run_one(
    name: str,
    *,
    time_limit: float,
    num_realizations: int,
    workers: int,
) -> dict:
    rel = f"{INSTANCE_REL}/{name}"
    t0 = time.time()
    jobShopEnvs = load_stochastic_job_shop_env(rel, num_realizations=num_realizations)
    model, vars_ = FJSPmodel_stoch.fjsp_stoch_cp_sat_model(
        jobShopEnvs, obj="mean", alpha=0.95
    )
    solver, status, solution_count = solve_model_workers(model, time_limit, workers)
    wall_solve = time.time() - t0
    status_name = solver.StatusName(status)
    row = {
        "instance": name.replace(".fjs", ""),
        "status": status_name,
        "cp_obj_search": None,
        "cp_wall_s": round(wall_solve, 3),
        "solution_count": solution_count,
        "eval_mean_cmax": None,
        "eval_var95": None,
        "num_realizations_cp": num_realizations,
        "time_limit_s": time_limit,
        "workers": workers,
        "error": "",
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        row["error"] = f"no feasible: {status_name}"
        return row

    jobShopEnvs, results_list = FJSPmodel_stoch.update_envs(
        jobShopEnvs, vars_, solver, status, solution_count, time_limit
    )
    results = results_list[0]
    row["cp_obj_search"] = float(results["objValue"])
    row["cp_wall_s"] = round(float(results["runtime"]), 3)

    inst = load_instance(INST_DIR / name)
    sol = schedule_to_solution(inst, results["Schedule"])
    pool = load_pool(REAL_DIR / name)
    eval_sc = ScenarioBundle(
        op_pt_scen=pool[EVAL_START : EVAL_START + EVAL_K],
        role="eval",
        source_slice=(EVAL_START, EVAL_START + EVAL_K),
    )
    metrics = evaluate_solution(inst, sol, eval_sc)
    row["eval_mean_cmax"] = float(metrics.mean_cmax)
    row["eval_var95"] = float(metrics.var_cmax)
    return row


def budget_label(time_limit: float) -> str:
    if time_limit <= 60:
        return "short"
    if time_limit < 3600:
        return "medium"
    return "long"


def main() -> None:
    args = parse_args()
    # FJSP-DRL params.py calls parse_args() at import time (via data_utils /
    # fjsp_loader). Keep only argv[0] after we have our Namespace.
    sys.argv = [sys.argv[0]]
    time_limit = float(args.time_limit)
    num_realizations = int(args.num_realizations)
    max_inst = int(args.max_inst)
    workers = int(args.workers)
    out_prefix = args.out_prefix.strip()
    out_csv = OUT_DIR / f"{out_prefix}-raw.csv"
    out_md = OUT_DIR / f"{out_prefix}.md"
    out_json = OUT_DIR / f"{out_prefix}-summary.json"
    label = budget_label(time_limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in REAL_DIR.glob("*.fjs"))[:max_inst]
    print(
        f"CP-stoch P0 ({label}): {len(names)} inst, tlim={time_limit}s, "
        f"K_cp={num_realizations}, workers={workers}, "
        f"eval=[{EVAL_START}:{EVAL_START+EVAL_K}], out={out_prefix}",
        flush=True,
    )
    rows = []
    done_ok = set()
    if args.resume and out_csv.exists():
        with out_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "FEASIBLE" and r.get("eval_mean_cmax") not in (
                    None,
                    "",
                ):
                    # normalize numeric fields
                    for k in (
                        "cp_obj_search",
                        "cp_wall_s",
                        "eval_mean_cmax",
                        "eval_var95",
                        "num_realizations_cp",
                        "time_limit_s",
                        "workers",
                    ):
                        if r.get(k) not in (None, ""):
                            try:
                                r[k] = float(r[k]) if "." in str(r[k]) or "e" in str(r[k]).lower() else int(float(r[k]))
                            except ValueError:
                                pass
                    if "solution_count" in r and r["solution_count"] not in (None, ""):
                        try:
                            r["solution_count"] = int(float(r["solution_count"]))
                        except ValueError:
                            pass
                    rows.append(r)
                    done_ok.add(r["instance"])
        print(
            f"Resume: loaded {len(done_ok)} FEASIBLE rows from {out_csv.name}",
            flush=True,
        )
    t0 = time.time()
    for i, name in enumerate(names):
        inst = name.replace(".fjs", "")
        if inst in done_ok:
            print(f"\n=== [{i+1}/{len(names)}] {name} SKIP (resume FEASIBLE) ===", flush=True)
            continue
        print(f"\n=== [{i+1}/{len(names)}] {name} ===", flush=True)
        try:
            row = run_one(
                name,
                time_limit=time_limit,
                num_realizations=num_realizations,
                workers=workers,
            )
        except Exception as e:
            row = {
                "instance": inst,
                "status": "ERROR",
                "cp_obj_search": None,
                "cp_wall_s": None,
                "solution_count": 0,
                "eval_mean_cmax": None,
                "eval_var95": None,
                "num_realizations_cp": num_realizations,
                "time_limit_s": time_limit,
                "workers": workers,
                "error": repr(e),
            }
            print("ERROR", e, flush=True)
        print(row, flush=True)
        # replace prior ERROR row for same instance if any
        rows = [r for r in rows if r.get("instance") != inst]
        rows.append(row)
        # keep instance order by names
        order = {n.replace(".fjs", ""): j for j, n in enumerate(names)}
        rows.sort(key=lambda r: order.get(r["instance"], 10**9))
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    feasible = [r for r in rows if r["eval_mean_cmax"] is not None]
    note = (
        f"CP-stoch official code (FJSPmodel_stoch mean). "
        f"Search on first K realizations; schedule decoded & re-eval "
        f"on P0 held-out [250:350] with SA-ALNS decoder. "
        f"Time limit {time_limit}s labeled **{label}-budget**"
        + (
            " — still << paper 3600s."
            if time_limit < 3600
            else " (paper-scale; report carefully)."
        )
    )
    summary = {
        "protocol": {
            "instances": len(names),
            "time_limit_s": time_limit,
            "num_realizations_cp": num_realizations,
            "workers": workers,
            "obj": "mean",
            "eval_slice": [EVAL_START, EVAL_START + EVAL_K],
            "budget_label": label,
            "out_prefix": out_prefix,
            "note": note,
        },
        "n_feasible": len(feasible),
        "n_failed": len(rows) - len(feasible),
        "macro_mean_cmax": float(np.mean([r["eval_mean_cmax"] for r in feasible]))
        if feasible
        else None,
        "macro_mean_var95": float(np.mean([r["eval_var95"] for r in feasible]))
        if feasible
        else None,
        "macro_cp_obj_search": float(np.mean([r["cp_obj_search"] for r in feasible]))
        if feasible
        else None,
        "mean_wall_s": float(
            np.mean([r["cp_wall_s"] for r in feasible if r["cp_wall_s"] is not None])
        )
        if feasible
        else None,
        "wall_total_s": time.time() - t0,
        "per_instance": rows,
        "spmdan_status": (
            "N/A — blocked: no pretrained SPM-DAN/DAN-stoch checkpoints in "
            "NCO-for-Stochastic-FJSP (GitHub releases empty for FJSP weights); "
            "short CPU train unfair vs paper and missing deps "
            "(gym, torch_geometric, einops) at probe time."
        ),
    }
    out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        f"# P0 CP-stoch ({label} budget {time_limit:g}s) vs SA-ALNS protocol",
        "",
        f"- Instances: {len(names)} × SD3 `10x5+mix`",
        f"- CP: OR-Tools CP-SAT stochastic **mean**, K_cp={num_realizations}, "
        f"time_limit={time_limit:g}s, workers={workers}",
        f"- Eval: re-decode CP assignment+sequence on scenarios "
        f"[{EVAL_START}:{EVAL_START+EVAL_K}] (same decoder as P0)",
        f"- Feasible: {len(feasible)}/{len(names)}",
        f"- Macro E[Cmax] (eval): {summary['macro_mean_cmax']}",
        f"- Macro VaR95 (eval): {summary['macro_mean_var95']}",
        f"- Macro CP search obj: {summary['macro_cp_obj_search']}",
        f"- Mean solve wall: {summary['mean_wall_s']}s; total {summary['wall_total_s']:.1f}s",
        f"- Label: **{label}-budget**; do not equate to paper 3600s"
        if time_limit < 3600
        else f"- Label: {label}-budget",
        "",
        "| Instance | status | CP search obj | eval E[Cmax] | VaR95 | wall_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance']} | {r['status']} | {r['cp_obj_search']} | "
            f"{r['eval_mean_cmax']} | {r['eval_var95']} | {r['cp_wall_s']} |"
        )
    lines += [
        "",
        "## SPM-DAN",
        summary["spmdan_status"],
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "per_instance"}, indent=2))


if __name__ == "__main__":
    main()
