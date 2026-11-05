"""Destroy operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from sfjsp_alns.decode import decode_schedule_times
from sfjsp_alns.types import Instance, ScenarioBundle, Solution


@dataclass
class DestroyResult:
    partial: Solution
    removed_ops: np.ndarray


class DestroyOp(Protocol):
    name: str

    def __call__(
        self,
        instance: Instance,
        solution: Solution,
        scenarios: ScenarioBundle | None,
        rng: np.random.Generator,
        q: float,
    ) -> DestroyResult: ...


def _n_remove(n: int, q: float) -> int:
    k = max(1, int(round(q * n)))
    return min(k, max(1, n // 2))


def _clear_machines(solution: Solution, removed: np.ndarray) -> Solution:
    new_mach = solution.machine_assign.copy()
    new_mach[removed] = -1
    return Solution(
        machine_assign=new_mach,
        op_sequence=solution.op_sequence.copy(),
        meta=dict(solution.meta),
    )


def random_ops(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle | None,
    rng: np.random.Generator,
    q: float,
) -> DestroyResult:
    """D1: clear machines for q fraction of ops; keep full sequence (M1-safe)."""
    n = instance.n_ops
    k = _n_remove(n, q)
    removed = rng.choice(n, size=k, replace=False).astype(np.int64)
    return DestroyResult(partial=_clear_machines(solution, removed), removed_ops=np.sort(removed))


def critical_path(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle | None,
    rng: np.random.Generator,
    q: float,
) -> DestroyResult:
    """D3: remove ops on a critical path under a sampled (or mean) scenario."""
    n = instance.n_ops
    k = _n_remove(n, q)
    if scenarios is None or scenarios.n_scenarios == 0:
        return random_ops(instance, solution, scenarios, rng, q)

    # Use one random search scenario for criticality (scenario-aware)
    s_idx = int(rng.integers(scenarios.n_scenarios))
    op_pt = scenarios.op_pt_scen[s_idx]
    starts, ends = decode_schedule_times(instance, solution, op_pt)
    cmax = float(ends.max())
    # backward critical: ops with end ≈ cmax chain via job/machine predecessors (approx by slack)
    slack = (cmax - ends) + starts * 0  # end-based: near-makespan ops
    # Prefer ops with end close to cmax and small forward slack
    score = ends.copy()
    # boost ops whose end is within 1e-6 of cmax, then walk predecessors loosely
    crit = np.flatnonzero(np.abs(ends - cmax) <= 1e-6 + 1e-9 * cmax)
    if crit.size == 0:
        crit = np.array([int(np.argmax(ends))], dtype=np.int64)
    # expand: include high-end ops ranked by end time
    order = np.argsort(-ends)
    picked: list[int] = []
    for op in list(crit) + list(order):
        op = int(op)
        if op not in picked:
            picked.append(op)
        if len(picked) >= k:
            break
    removed = np.array(picked[:k], dtype=np.int64)
    # if too few, pad with random
    if removed.size < k:
        rest = np.setdiff1d(np.arange(n), removed)
        extra = rng.choice(rest, size=k - removed.size, replace=False)
        removed = np.concatenate([removed, extra])
    return DestroyResult(partial=_clear_machines(solution, removed), removed_ops=np.sort(removed))


def high_var_ops(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle | None,
    rng: np.random.Generator,
    q: float,
) -> DestroyResult:
    """D4: remove ops with highest cross-scenario processing-time CV on assigned machine."""
    n = instance.n_ops
    k = _n_remove(n, q)
    if scenarios is None or scenarios.n_scenarios < 2:
        return random_ops(instance, solution, scenarios, rng, q)

    cvs = np.zeros(n, dtype=np.float64)
    for op in range(n):
        m = int(solution.machine_assign[op])
        if m < 0:
            cvs[op] = 0.0
            continue
        pts = scenarios.op_pt_scen[:, op, m].astype(np.float64)
        mu = float(pts.mean())
        if mu <= 1e-9:
            cvs[op] = 0.0
        else:
            cvs[op] = float(pts.std() / mu)
    # break ties randomly
    noise = rng.random(n) * 1e-9
    order = np.argsort(-(cvs + noise))
    removed = order[:k].astype(np.int64)
    return DestroyResult(partial=_clear_machines(solution, removed), removed_ops=np.sort(removed))


def machine_load(
    instance: Instance,
    solution: Solution,
    scenarios: ScenarioBundle | None,
    rng: np.random.Generator,
    q: float,
) -> DestroyResult:
    """D5: clear ops on the heaviest-loaded machines (mean scenario load)."""
    n = instance.n_ops
    k = _n_remove(n, q)
    n_m = instance.n_machines
    loads = np.zeros(n_m, dtype=np.float64)
    if scenarios is not None and scenarios.n_scenarios > 0:
        mean_pt = scenarios.op_pt_scen.mean(axis=0)
    else:
        mean_pt = instance.op_pt_nom.astype(np.float64)
    for op in range(n):
        m = int(solution.machine_assign[op])
        if m >= 0:
            loads[m] += float(mean_pt[op, m])
    # ops ranked by their machine's load
    op_score = np.zeros(n, dtype=np.float64)
    for op in range(n):
        m = int(solution.machine_assign[op])
        op_score[op] = loads[m] if m >= 0 else 0.0
    noise = rng.random(n) * 1e-9
    order = np.argsort(-(op_score + noise))
    removed = order[:k].astype(np.int64)
    return DestroyResult(partial=_clear_machines(solution, removed), removed_ops=np.sort(removed))


random_ops.name = "random_ops"  # type: ignore[attr-defined]
critical_path.name = "critical_path"  # type: ignore[attr-defined]
high_var_ops.name = "high_var_ops"  # type: ignore[attr-defined]
machine_load.name = "machine_load"  # type: ignore[attr-defined]

DESTROY_OPS: dict[str, DestroyOp] = {
    "random_ops": random_ops,  # type: ignore[dict-item]
    "critical_path": critical_path,  # type: ignore[dict-item]
    "high_var_ops": high_var_ops,  # type: ignore[dict-item]
    "machine_load": machine_load,  # type: ignore[dict-item]
}


def _make_not_impl(name: str) -> Callable:
    def _fn(*_a, **_k):
        raise NotImplementedError(name)

    _fn.name = name  # type: ignore[attr-defined]
    return _fn


DESTROY_OPS["random_seq"] = _make_not_impl("D2:random_seq")  # type: ignore[assignment]
DESTROY_OPS["job_block"] = _make_not_impl("D6:job_block")  # type: ignore[assignment]
