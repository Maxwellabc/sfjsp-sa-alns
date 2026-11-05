"""Scenario aggregation: mean / VaR over ScenarioBundle."""

from __future__ import annotations

from typing import Literal

import numpy as np

from sfjsp_alns.decode import decode_cmax
from sfjsp_alns.types import Instance, Metrics, ScenarioBundle, Solution


def evaluate_solution(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle,
    *,
    obj: Literal["mean", "VaR"] = "mean",
    alpha: float = 0.95,
) -> Metrics:
    cmaxs = np.empty(scenarios.n_scenarios, dtype=np.float64)
    for k in range(scenarios.n_scenarios):
        cmaxs[k] = decode_cmax(instance, solution, scenarios.op_pt_scen[k])
    mean = float(cmaxs.mean())
    var = float(np.percentile(cmaxs, 100.0 * alpha))
    return Metrics(
        mean_cmax=mean,
        var_cmax=var,
        alpha=alpha,
        n_scenarios=scenarios.n_scenarios,
        decode_count=scenarios.n_scenarios,
    )


def objective_value(metrics: Metrics, obj: Literal["mean", "VaR"] = "mean") -> float:
    return metrics.mean_cmax if obj == "mean" else metrics.var_cmax


def mean_cmax_fast(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle,
    idx: np.ndarray | None = None,
) -> float:
    """Mean Cmax over all or a subset of scenario indices."""
    if idx is None:
        idx = np.arange(scenarios.n_scenarios)
    total = 0.0
    for k in idx:
        total += decode_cmax(instance, solution, scenarios.op_pt_scen[int(k)])
    return total / max(len(idx), 1)
