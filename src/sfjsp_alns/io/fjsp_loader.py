"""Load deterministic ``.fjs`` files and Smit stochastic realization banks.

The parser is in-repo (``fjs_text``). It follows FJSP-DRL
``load_data_from_stochastic_realizations_file``: scenario index 0 is the first
block *after* the deterministic header, which is the slice Smit evaluates at
``[250 : 250 + K_eval]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sfjsp_alns.io.fjs_text import read_instance_text, read_realization_matrices
from sfjsp_alns.types import Instance, ScenarioBundle


def load_instance(path: str | Path) -> Instance:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    job_length, op_pt = read_instance_text(path)
    return Instance(
        name=path.stem,
        job_length=np.asarray(job_length, dtype=np.int64),
        op_pt_nom=np.asarray(op_pt, dtype=np.int64),
        path=str(path),
    )


def load_realization_pool(path: str | Path, *, max_realizations: int = 10**9) -> np.ndarray:
    """Stack stochastic scenarios as ``(K, N, M)`` int64. Nominal block excluded."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mats = read_realization_matrices(path, max_realizations=max_realizations)
    return np.stack(mats, axis=0)


def load_scenario_bundle(
    path: str | Path,
    *,
    start: int = 250,
    count: int = 50,
    role: str = "eval",
    max_realizations: int = 1250,
) -> ScenarioBundle:
    """Slice scenarios ``[start:start+count]`` from a realizations file.

    ``max_realizations`` is the FJSP-DRL cap (that loader keeps
    ``max_realizations + 1`` blocks). The default 1250 is enough for the
    paper slice ``[250:1250]`` (K_eval=1000). Raise it if you need a longer bank.
    """
    if start < 0 or count < 1:
        raise ValueError("start must be >= 0 and count must be >= 1")
    pool = load_realization_pool(path, max_realizations=max(max_realizations, start + count - 1))
    end = start + count
    if end > pool.shape[0]:
        raise ValueError(
            f"need realizations [{start}:{end}] ({count} scenarios) from {path}, "
            f"but the file yielded {pool.shape[0]} stochastic blocks. "
            "Regenerate with NUM_SAMPLES >= eval_start + k_eval "
            "(1250 for K_eval=1000 at eval_start=250)."
        )
    return ScenarioBundle(
        op_pt_scen=pool[start:end],
        role=role,  # type: ignore[arg-type]
        source_slice=(start, end),
    )
