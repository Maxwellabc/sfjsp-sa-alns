"""Decoder and PDR smoke on a shipped tiny fixture."""

from __future__ import annotations

import numpy as np

from sfjsp_alns.decode import decode_cmax
from sfjsp_alns.evaluate import evaluate_solution
from sfjsp_alns.init_pdr import pdr_solution
from sfjsp_alns.io.fjsp_loader import load_instance
from sfjsp_alns.k_eval import synthetic_pool
from sfjsp_alns.paths import repo_root
from sfjsp_alns.types import ScenarioBundle


def test_decode_finite_on_synthetic_fixture():
    inst = load_instance(repo_root() / "data" / "fixtures" / "synthetic" / "tiny_001.fjs")
    sol = pdr_solution(inst, "MOR", seed=50)
    c = decode_cmax(inst, sol, inst.op_pt_nom)
    assert np.isfinite(c) and c > 0

    pool = synthetic_pool(inst, n_scenarios=8, seed=50)
    scen = ScenarioBundle(op_pt_scen=pool, role="eval", source_slice=(0, 8))
    metrics = evaluate_solution(inst, sol, scen)
    assert metrics.n_scenarios == 8
    assert metrics.mean_cmax > 0
    assert metrics.var_cmax >= metrics.mean_cmax * 0.5
