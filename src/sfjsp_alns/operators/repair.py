"""Repair operators."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from sfjsp_alns.decode import decode_cmax
from sfjsp_alns.instance_utils import first_last_op, op_to_job
from sfjsp_alns.operators.destroy import DestroyResult
from sfjsp_alns.types import Instance, ScenarioBundle, Solution


class RepairOp(Protocol):
    name: str

    def __call__(
        self,
        instance: Instance,
        destroy: DestroyResult,
        scenarios: ScenarioBundle,
        rng: np.random.Generator,
        obj: str = "mean",
    ) -> Solution: ...


def _scenario_mean_pt(instance: Instance, scenarios: ScenarioBundle) -> np.ndarray:
    mean_pt = scenarios.op_pt_scen.mean(axis=0).copy()
    mean_pt[instance.op_pt_nom <= 0] = 0
    return mean_pt


def spt_biased(
    instance: Instance,
    destroy: DestroyResult,
    scenarios: ScenarioBundle,
    rng: np.random.Generator,
    obj: str = "mean",
) -> Solution:
    """
    R4: for each cleared op, assign SPT-biased machine; optionally try a few
    precedence-preserving relocate moves for that op in the sequence.
    """
    mean_pt = _scenario_mean_pt(instance, scenarios)
    first, last = first_last_op(instance.job_length)
    job_of = op_to_job(instance.job_length)
    seq = destroy.partial.op_sequence.copy()
    mach = destroy.partial.machine_assign.copy()
    removed = list(destroy.removed_ops.tolist())
    rng.shuffle(removed)

    k_fast = min(8, scenarios.n_scenarios)
    idx = rng.choice(scenarios.n_scenarios, size=k_fast, replace=False)

    def score(sol: Solution) -> float:
        total = 0.0
        for k in idx:
            total += decode_cmax(instance, sol, scenarios.op_pt_scen[int(k)])
        return total / k_fast

    # first pass: fill all machines with SPT (deterministic greedy)
    for op in removed:
        op = int(op)
        comps = np.flatnonzero(instance.op_pt_nom[op] > 0)
        pts = mean_pt[op, comps]
        mach[op] = int(comps[int(np.argmin(pts))])

    cur = Solution(machine_assign=mach.copy(), op_sequence=seq.copy())
    cur_sc = score(cur)

    # second pass: for each removed op try top-2 machines + a few relocate positions
    for op in removed:
        op = int(op)
        comps = np.flatnonzero(instance.op_pt_nom[op] > 0)
        pts = mean_pt[op, comps]
        order = np.argsort(pts)
        cand_m = [int(comps[i]) for i in order[: min(3, len(order))]]

        # remove op from sequence temporarily
        pos_old = int(np.where(seq == op)[0][0])
        seq_wo = np.delete(seq, pos_old)
        j = int(job_of[op])
        pred = op - 1 if op > first[j] else None
        succ = op + 1 if op < last[j] else None
        pos_pred = -1
        pos_succ = len(seq_wo)
        for i, x in enumerate(seq_wo):
            if pred is not None and int(x) == pred:
                pos_pred = i
            if succ is not None and int(x) == succ:
                pos_succ = i
        positions = list(range(pos_pred + 1, pos_succ + 1))
        if len(positions) > 5:
            pick = {
                positions[0],
                positions[len(positions) // 2],
                positions[-1],
                int(rng.choice(positions)),
                pos_old if pos_old <= len(seq_wo) else positions[-1],
            }
            positions = sorted(p for p in pick if pos_pred + 1 <= p <= pos_succ)

        best_seq, best_mach, best_sc = seq, mach, cur_sc
        for m in cand_m:
            for pos in positions:
                new_seq = np.insert(seq_wo, pos, op)
                new_mach = mach.copy()
                new_mach[op] = m
                sol = Solution(machine_assign=new_mach, op_sequence=new_seq)
                sc = score(sol)
                if sc + 1e-9 < best_sc:
                    best_sc = sc
                    best_seq = new_seq
                    best_mach = new_mach
        seq, mach, cur_sc = best_seq, best_mach, best_sc

    return Solution(machine_assign=mach, op_sequence=seq, meta=dict(destroy.partial.meta))


def _not_impl(name: str) -> Callable:
    def _fn(*_a, **_k):
        raise NotImplementedError(name)

    _fn.name = name  # type: ignore[attr-defined]
    return _fn


spt_biased.name = "spt_biased"  # type: ignore[attr-defined]

REPAIR_OPS: dict[str, RepairOp] = {
    "greedy_mean": _not_impl("R1:greedy_mean"),  # type: ignore[dict-item]
    "greedy_var": _not_impl("R2:greedy_var"),  # type: ignore[dict-item]
    "regret": _not_impl("R3:regret"),  # type: ignore[dict-item]
    "spt_biased": spt_biased,  # type: ignore[dict-item]
    "beam": _not_impl("R5:beam"),  # type: ignore[dict-item]
}
