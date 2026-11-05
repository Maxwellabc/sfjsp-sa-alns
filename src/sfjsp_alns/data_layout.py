"""Where external SD3 banks live. This repo does not ship them."""

from __future__ import annotations

from pathlib import Path

EXTERNAL_DATA_HELP = """
Full SD3 instance and realization banks are not in this seed. Clone

  git clone https://github.com/ai-for-decision-making-tue/NCO-for-Stochastic-FJSP.git

and point --data-root at the FJSP-DRL data directory (either of):

  <NCO>/FJSP-DRL/data
  <NCO>                  (accepted if FJSP-DRL/data/SD3 exists)
  <NCO>/FJSP-DRL         (accepted if data/SD3 exists)

Expected layout for size 10x5+mix (lognormal random-CoV; no extra dist folder):

  <data-root>/SD3/10x5+mix/*.fjs
  <data-root>/stochastic_realizations/random/SD3/10x5+mix/*.fjs

Paper K_eval=1000 reads scenarios [250:1250], so each realizations file needs
at least 1250 stochastic blocks (NUM_SAMPLES>=1250). P0/full-100 in this track
used NUM_SAMPLES=450 and K_eval=100 ([250:350]) and cannot serve K_eval=1000
until the bank is regenerated.

CP-stoch additionally reads the benchmark tree, not FJSP-DRL/data:

  <NCO>/Job_Shop_Scheduling_Benchmark_Environments_and_Instances/data/fjsp/stoch/random/SD3/10x5+mix/*.fjs

Pass that checkout as --nco-root (default: baselines/NCO-for-Stochastic-FJSP).
SPM-DAN weights are not published; do not expect a checkpoint under FJSP-DRL.
""".strip()


def resolve_fjsp_drl_data(path: str | Path) -> Path:
    """Return the directory that directly contains ``SD3/``."""
    raw = Path(path).expanduser()
    candidates = [
        raw,
        raw / "data",
        raw / "FJSP-DRL" / "data",
        raw / "baselines" / "NCO-for-Stochastic-FJSP" / "FJSP-DRL" / "data",
    ]
    for cand in candidates:
        if (cand / "SD3").is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        f"{raw} is not an FJSP-DRL data root (no SD3/ directory).\n{EXTERNAL_DATA_HELP}"
    )


def sd3_instance_dir(data_root: Path, size: str) -> Path:
    return data_root / "SD3" / size


def sd3_realization_dir(data_root: Path, size: str) -> Path:
    return data_root / "stochastic_realizations" / "random" / "SD3" / size
