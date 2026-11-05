"""CLI entry: python -m sfjsp_alns.cli --help (wires after M1)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scenario-aware ALNS for Stochastic FJSP")
    p.add_argument("--instance", type=str, required=False, help="Path to .fjs")
    p.add_argument("--realizations", type=str, required=False, help="Path to realizations .fjs")
    p.add_argument("--eval-start", type=int, default=250)
    p.add_argument("--eval-count", type=int, default=50)
    p.add_argument("--obj", choices=["mean", "VaR"], default="mean")
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--time-limit", type=float, default=60.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        "Held-out eval: python scripts/alns_k_eval_1000.py --help\n"
        "P0 / full-100 ALNS and CP-stoch flags are documented in README.md.\n"
        "SPM-DAN weights are N/A."
    )
    print(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
