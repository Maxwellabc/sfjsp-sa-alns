#!/usr/bin/env python3
"""Held-out K_eval evaluation for SA-ALNS (paper default K_eval=1000).

Smoke (no external SD3 bank)::

    python scripts/alns_k_eval_1000.py \\
        --synthetic --instances 2 --k-eval 8 --eval-start 20 --k-search 8 \\
        --max-iter 15 --time-limit 2 --seed 50 \\
        --out results/k-eval-smoke.json

Full job (external bank; long)::

    python scripts/alns_k_eval_1000.py \\
        --data-root /path/to/FJSP-DRL/data \\
        --instances 100 --k-eval 1000 --seeds 50,51,52 \\
        --out results/k-eval-1000-summary.json

Synthetic output is a wiring check. It is not an SD3 table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfjsp_alns.data_layout import (  # noqa: E402
    EXTERNAL_DATA_HELP,
    resolve_fjsp_drl_data,
    sd3_instance_dir,
    sd3_realization_dir,
)
from sfjsp_alns.io.fjsp_loader import load_instance, load_realization_pool  # noqa: E402
from sfjsp_alns.k_eval import (  # noqa: E402
    KEvalSpec,
    aggregate,
    assert_slices_disjoint,
    evaluate_instance,
    synthetic_pool,
)
from sfjsp_alns.paths import repo_root  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Out-of-sample SA-ALNS evaluation (default K_eval=1000)."
    )
    p.add_argument(
        "--instances",
        type=int,
        default=1,
        help="How many instances (first N, sorted by filename). Smoke: 1 or 2.",
    )
    p.add_argument(
        "--k-eval",
        type=int,
        default=1000,
        help="Held-out scenario count. Paper protocol is 1000.",
    )
    p.add_argument("--seed", type=int, default=50, help="Solver seed, and synthetic-pool seed.")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated solver seeds (overrides a single --seed). Example: 50,51,52.",
    )
    p.add_argument("--out", type=Path, required=True, help="JSON summary path.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="External FJSP-DRL data directory (SD3/ and stochastic_realizations/). Not shipped.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use data/fixtures/synthetic and an in-process scenario draw. Not paper numbers.",
    )
    p.add_argument("--size", type=str, default="10x5+mix", help="SD3 size folder under --data-root.")
    p.add_argument("--eval-start", type=int, default=250, help="First held-out scenario index.")
    p.add_argument("--k-search", type=int, default=50, help="Search scenario count.")
    p.add_argument("--search-start", type=int, default=0, help="First search scenario index.")
    p.add_argument("--max-iter", type=int, default=None, help="ALNS iterations (default 40 synthetic / 300 external).")
    p.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="ALNS seconds per seed (default 5 synthetic / 40 external).",
    )
    return p


def _seeds(args: argparse.Namespace) -> tuple[int, ...]:
    if args.seeds:
        seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
        if not seeds:
            raise SystemExit("--seeds was empty")
        return seeds
    return (int(args.seed),)


def _jobs_synthetic(n: int, spec: KEvalSpec, scenario_seed: int):
    fixture_dir = repo_root() / "data" / "fixtures" / "synthetic"
    files = sorted(fixture_dir.glob("*.fjs"))[:n]
    if len(files) < n:
        raise SystemExit(
            f"Need {n} synthetic fixtures in {fixture_dir}, found {len(files)}."
        )
    jobs = []
    for path in files:
        inst = load_instance(path)
        pool = synthetic_pool(inst, spec.pool_length, scenario_seed)
        jobs.append((inst, pool))
    return jobs


def _jobs_external(n: int, spec: KEvalSpec, data_root: Path, size: str):
    try:
        root = resolve_fjsp_drl_data(data_root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    inst_dir = sd3_instance_dir(root, size)
    real_dir = sd3_realization_dir(root, size)
    if not inst_dir.is_dir() or not real_dir.is_dir():
        raise SystemExit(
            f"Missing SD3 folders under {root} for size {size}.\n{EXTERNAL_DATA_HELP}"
        )
    files = sorted(inst_dir.glob("*.fjs"))[:n]
    if len(files) < n:
        raise SystemExit(
            f"Requested --instances {n} but {inst_dir} has {len(files)} .fjs files."
        )
    jobs = []
    for path in files:
        real = real_dir / path.name
        if not real.is_file():
            raise SystemExit(
                f"No realization bank for {path.name} at {real}.\n{EXTERNAL_DATA_HELP}"
            )
        inst = load_instance(path)
        pool = load_realization_pool(real, max_realizations=max(spec.pool_length, 1))
        if pool.shape[0] < spec.pool_length:
            raise SystemExit(
                f"{real.name} has {pool.shape[0]} stochastic blocks; "
                f"need {spec.pool_length} for eval [{spec.eval_start}:{spec.eval_stop}] "
                f"and search [{spec.search_start}:{spec.search_stop}]. "
                "Regenerate with scripts/generate_sd3_10x5_realizations.py "
                "--num-samples >= eval_start+k_eval (1250 for K_eval=1000).\n"
                f"{EXTERNAL_DATA_HELP}"
            )
        jobs.append((inst, pool))
    return jobs, root


def _write_outputs(out: Path, summary: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    rows = payload.pop("rows")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "instance",
            "method",
            "seed",
            "mean_cmax",
            "var95",
            "iters",
            "seconds",
            "n_eval",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path = out.with_suffix(".md")
    proto = summary["protocol"]
    lines = [
        "# K_eval held-out SA-ALNS",
        "",
        f"- kind: `{proto['kind']}`",
        f"- comparable_to_paper_protocol: **{proto['comparable_to_paper_protocol']}**",
        f"- instances: {summary['n_instances']}",
        f"- eval slice: `{proto['eval_slice']}` (K_eval={proto['k_eval']})",
        f"- search slice: `{proto['search_slice']}`",
        f"- seeds: {proto['seeds']}",
        f"- ALNS: max_iter={proto['max_iter']}, time_limit_s={proto['time_limit_s']}",
        "",
        proto["note"],
        "",
        "| Method | E[Cmax] | VaR95 |",
        "|---|---:|---:|",
    ]
    for method, value in summary["macro_mean_cmax"].items():
        var = summary["macro_mean_var95"][method]
        lines.append(f"| {method} | {value:.4f} | {var:.4f} |")
    lines += ["", summary["spmdan"], ""]
    if not proto["comparable_to_paper_protocol"]:
        lines.append(
            "Do not cite this file as the paper K_eval=1000 SD3 result. "
            "That run requires `--data-root` with banks of at least 1250 scenarios."
        )
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.instances < 1:
        raise SystemExit("--instances must be >= 1")
    if args.data_root is not None and args.synthetic:
        raise SystemExit("Pass only one of --data-root and --synthetic.")
    synthetic = args.synthetic or args.data_root is None
    if args.data_root is None and not args.synthetic:
        print(
            "No --data-root given; using synthetic fixtures. "
            "This is not an SD3 result. Pass --data-root for the external bank.",
            file=sys.stderr,
        )
    seeds = _seeds(args)
    max_iter = args.max_iter if args.max_iter is not None else (40 if synthetic else 300)
    time_limit = args.time_limit if args.time_limit is not None else (5.0 if synthetic else 40.0)
    spec = KEvalSpec(
        k_eval=args.k_eval,
        eval_start=args.eval_start,
        k_search=args.k_search,
        search_start=args.search_start,
        seeds=seeds,
        max_iter=max_iter,
        time_limit_s=time_limit,
        kind="synthetic_fixture" if synthetic else "external_sd3",
    )
    try:
        assert_slices_disjoint(spec)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    data_root_used = None
    if synthetic:
        jobs = _jobs_synthetic(args.instances, spec, args.seed)
    else:
        jobs, data_root_used = _jobs_external(args.instances, spec, args.data_root, args.size)

    print(
        f"K_eval kind={spec.kind} instances={len(jobs)} "
        f"eval=[{spec.eval_start}:{spec.eval_stop}] "
        f"search=[{spec.search_start}:{spec.search_stop}] "
        f"seeds={list(spec.seeds)} max_iter={spec.max_iter} tlim={spec.time_limit_s}s",
        flush=True,
    )
    rows = []
    for inst, pool in jobs:
        print(f"  {inst.name} pool={pool.shape[0]}", flush=True)
        try:
            inst_rows = evaluate_instance(inst, pool, spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        rows.extend(inst_rows)
        alns = [r for r in inst_rows if r["method"] == "SA-ALNS"]
        for row in alns:
            print(
                f"    seed={row['seed']} SA-ALNS E={row['mean_cmax']:.2f} "
                f"VaR={row['var95']:.2f} iters={row['iters']} {row['seconds']:.1f}s",
                flush=True,
            )
    summary = aggregate(rows, spec, n_instances=len(jobs))
    summary["protocol"]["data_root"] = None if data_root_used is None else str(data_root_used)
    summary["protocol"]["size"] = None if synthetic else args.size
    summary["protocol"]["scenario_seed"] = int(args.seed) if synthetic else None
    _write_outputs(args.out, summary)
    print("macro E[Cmax]:", summary["macro_mean_cmax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
