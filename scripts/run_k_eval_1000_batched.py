#!/usr/bin/env python3
"""Batched K_eval=1000 driver: partial summaries after each batch, final merge.

Does not touch P0 banks. Intended for use alongside a long CP job (nice the process).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import alns_k_eval_1000 as harness  # noqa: E402
from sfjsp_alns.k_eval import KEvalSpec, aggregate, assert_slices_disjoint, evaluate_instance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--size", type=str, default="10x5+mix")
    p.add_argument("--instances", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--k-eval", type=int, default=1000)
    p.add_argument("--seeds", type=str, default="50,51,52")
    p.add_argument("--eval-start", type=int, default=250)
    p.add_argument("--k-search", type=int, default=50)
    p.add_argument("--search-start", type=int, default=0)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--time-limit", type=float, default=40.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already present in out.with_suffix('.rows.json')",
    )
    return p


def _load_done(rows_path: Path) -> set[str]:
    if not rows_path.is_file():
        return set()
    data = json.loads(rows_path.read_text(encoding="utf-8"))
    return {r["instance"] for r in data.get("rows", [])}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    spec = KEvalSpec(
        k_eval=args.k_eval,
        eval_start=args.eval_start,
        k_search=args.k_search,
        search_start=args.search_start,
        seeds=seeds,
        max_iter=args.max_iter,
        time_limit_s=args.time_limit,
        kind="external_sd3",
    )
    assert_slices_disjoint(spec)
    jobs, data_root_used = harness._jobs_external(
        args.instances, spec, args.data_root, args.size
    )

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_path = out.with_suffix(".rows.json")
    rows: list[dict] = []
    if args.resume and rows_path.is_file():
        rows = list(json.loads(rows_path.read_text(encoding="utf-8")).get("rows", []))
        print(f"resume: loaded {len(rows)} rows from {rows_path}", flush=True)
    done = {r["instance"] for r in rows}

    print(
        f"batched K_eval instances={len(jobs)} batch={args.batch_size} "
        f"eval=[{spec.eval_start}:{spec.eval_stop}] seeds={list(seeds)} "
        f"max_iter={spec.max_iter} tlim={spec.time_limit_s}s",
        flush=True,
    )

    pending = [(inst, pool) for inst, pool in jobs if inst.name not in done]
    batch_idx = 0
    for i in range(0, len(pending), args.batch_size):
        batch = pending[i : i + args.batch_size]
        batch_idx += 1
        print(f"=== batch {batch_idx}: {len(batch)} instances ===", flush=True)
        for inst, pool in batch:
            print(f"  {inst.name} pool={pool.shape[0]}", flush=True)
            inst_rows = evaluate_instance(inst, pool, spec)
            rows.extend(inst_rows)
            for row in inst_rows:
                if row["method"] == "SA-ALNS":
                    print(
                        f"    seed={row['seed']} SA-ALNS E={row['mean_cmax']:.2f} "
                        f"VaR={row['var95']:.2f} iters={row['iters']} {row['seconds']:.1f}s",
                        flush=True,
                    )
        rows_path.write_text(
            json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
        )
        n_inst = len({r["instance"] for r in rows})
        summary = aggregate(rows, spec, n_instances=n_inst)
        summary["protocol"]["data_root"] = str(data_root_used)
        summary["protocol"]["size"] = args.size
        summary["protocol"]["scenario_seed"] = None
        summary["protocol"]["batch_note"] = (
            f"partial/complete after batch {batch_idx}; "
            f"{n_inst}/{len(jobs)} instances in summary"
        )
        harness._write_outputs(out, summary)
        print(f"macro E[Cmax] after {n_inst} inst:", summary["macro_mean_cmax"], flush=True)

    print("batched K_eval done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
