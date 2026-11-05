#!/usr/bin/env python3
"""Generate lognormal random-CoV realization banks for SD3 (official recipe).

Does **not** import FJSP-DRL. Uses in-repo ``fjs_text`` + ``scenarios``.
Walks instances in sorted order on NumPy's global RNG after ``np.random.seed``.

P0/full-100: ``--num-samples 450``. Paper K_eval=1000: ``--num-samples 1250``.
Never overwrite an existing shorter P0 bank in place — point ``--data-root`` at a
parallel tree (symlink SD3/, write new stochastic_realizations/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfjsp_alns.data_layout import resolve_fjsp_drl_data, sd3_instance_dir, sd3_realization_dir
from sfjsp_alns.io.fjs_text import read_instance_text, write_stochastic_realizations_file
from sfjsp_alns.scenarios import lognormal_random_cov_bank


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="FJSP-DRL data directory containing SD3/ (or a parallel k1250 root).",
    )
    p.add_argument("--size", type=str, default="10x5+mix")
    p.add_argument("--limit", type=int, default=20, help="First N sorted instances")
    p.add_argument(
        "--names",
        type=str,
        default="",
        help="Comma-separated filenames; overrides --limit when set",
    )
    p.add_argument("--num-samples", type=int, default=450)
    p.add_argument("--seed", type=int, default=200, help="seed_datagen (official default 200)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing realizations file",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    root = resolve_fjsp_drl_data(args.data_root)
    inst_dir = sd3_instance_dir(root, args.size)
    out_dir = sd3_realization_dir(root, args.size)
    if not inst_dir.is_dir():
        raise SystemExit(f"missing instance dir {inst_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.names:
        files = [inst_dir / n.strip() for n in args.names.split(",") if n.strip()]
        for f in files:
            if not f.is_file():
                raise SystemExit(f"missing instance {f}")
    else:
        files = sorted(inst_dir.glob("*.fjs"))[: int(args.limit)]
    if not files:
        raise SystemExit(f"no .fjs under {inst_dir}")

    rng_state = np.random.get_state()
    np.random.seed(int(args.seed))
    print(
        f"Generating {len(files)} × {args.num_samples} → {out_dir} (seed={args.seed})",
        flush=True,
    )
    for path in files:
        out = out_dir / path.name
        if out.exists() and not args.force:
            raise SystemExit(
                f"refusing to overwrite {out} (pass --force, or use a parallel "
                f"--data-root for the k1250 bank so the P0/450 bank stays intact)"
            )
        job_length, op_pt = read_instance_text(path)
        samples = lognormal_random_cov_bank(op_pt, int(args.num_samples))
        write_stochastic_realizations_file(out, job_length, op_pt, samples)
        print(f"  wrote {out.name} K={samples.shape[0]}", flush=True)
    np.random.set_state(rng_state)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
