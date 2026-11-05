"""Priority dispatching rules -> Solution (standalone, no torch env)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from sfjsp_alns.instance_utils import first_last_op
from sfjsp_alns.types import Instance, Solution

PDRName = Literal["FIFO", "MOR", "SPT", "MWKR"]


def pdr_solution(
    instance: Instance,
    method: PDRName = "MOR",
    *,
    op_pt: np.ndarray | None = None,
    seed: int = 50,
) -> Solution:
    """
    Roll out a PDR on a processing-time matrix (default: nominal).

    Scheduling policy mirrors Smit FJSP-DRL heuristics at a high level:
    ready jobs (job_ready <= next_event_time) compete; machine chosen for
    earliest feasible start among compatible machines.
    """
    rng = np.random.default_rng(seed)
    pt = instance.op_pt_nom if op_pt is None else op_pt
    first, last = first_last_op(instance.job_length)
    n_jobs, n_m, n_ops = instance.n_jobs, instance.n_machines, instance.n_ops

    next_op = first.copy()
    job_ready = np.zeros(n_jobs, dtype=np.float64)
    mch_ready = np.zeros(n_m, dtype=np.float64)
    machine_assign = np.full(n_ops, -1, dtype=np.int64)
    op_sequence = np.empty(n_ops, dtype=np.int64)
    placed = 0

    # remaining work using mean compatible pt
    def remain_ops(j: int) -> int:
        return int(last[j] - next_op[j] + 1) if next_op[j] <= last[j] else 0

    def remain_work(j: int) -> float:
        if next_op[j] > last[j]:
            return 0.0
        w = 0.0
        for op in range(next_op[j], last[j] + 1):
            row = pt[op]
            comp = row[row > 0]
            w += float(comp.mean()) if len(comp) else 0.0
        return w

    while placed < n_ops:
        unfinished = [j for j in range(n_jobs) if next_op[j] <= last[j]]
        # next schedule time = min over ready candidates of max(job_ready, min mch ready)
        # approximate official next_schedule_time: min job_ready among unfinished
        # then allow jobs with job_ready <= t_now where t_now is min start among feasible
        # Simpler active set: jobs that can start at or before the earliest possible start
        earliest_starts = []
        for j in unfinished:
            op = int(next_op[j])
            comps = np.flatnonzero(pt[op] > 0)
            if len(comps) == 0:
                raise ValueError(f"op {op} has no machine")
            best = min(max(job_ready[j], mch_ready[m]) for m in comps)
            earliest_starts.append(best)
        t_now = float(min(earliest_starts))
        ready_jobs = [j for j in unfinished if job_ready[j] <= t_now + 1e-9]
        if not ready_jobs:
            ready_jobs = unfinished

        if method == "FIFO":
            vals = np.array([job_ready[j] for j in ready_jobs])
            cand = [ready_jobs[i] for i in np.where(vals == vals.min())[0]]
            j = int(rng.choice(cand))
            op = int(next_op[j])
            comps = np.flatnonzero(pt[op] > 0)
            m_times = np.array([mch_ready[m] for m in comps])
            m = int(comps[rng.choice(np.where(m_times == m_times.min())[0])])
        elif method == "MOR":
            vals = np.array([remain_ops(j) for j in ready_jobs])
            cand = [ready_jobs[i] for i in np.where(vals == vals.max())[0]]
            j = int(rng.choice(cand))
            op = int(next_op[j])
            m = _pick_immediate_machine(op, j, pt, job_ready, mch_ready, rng)
        elif method == "MWKR":
            vals = np.array([remain_work(j) for j in ready_jobs])
            cand = [ready_jobs[i] for i in np.where(vals == vals.max())[0]]
            j = int(rng.choice(cand))
            op = int(next_op[j])
            m = _pick_immediate_machine(op, j, pt, job_ready, mch_ready, rng)
        elif method == "SPT":
            # among ready job-machine pairs, shortest pt
            best_pt = np.inf
            pairs = []
            for j in ready_jobs:
                op = int(next_op[j])
                for m in np.flatnonzero(pt[op] > 0):
                    ptv = int(pt[op, m])
                    if ptv < best_pt:
                        best_pt = ptv
                        pairs = [(j, op, int(m))]
                    elif ptv == best_pt:
                        pairs.append((j, op, int(m)))
            j, op, m = pairs[int(rng.integers(len(pairs)))]
        else:
            raise ValueError(method)

        start = max(job_ready[j], mch_ready[m])
        end = start + int(pt[op, m])
        job_ready[j] = end
        mch_ready[m] = end
        machine_assign[op] = m
        op_sequence[placed] = op
        placed += 1
        next_op[j] += 1

    return Solution(
        machine_assign=machine_assign,
        op_sequence=op_sequence,
        meta={"init": method, "seed": seed},
    )


def _pick_immediate_machine(op, j, pt, job_ready, mch_ready, rng) -> int:
    comps = np.flatnonzero(pt[op] > 0)
    jf = job_ready[j]
    mf = np.array([mch_ready[m] for m in comps])
    if np.all(jf < mf):
        chosen = comps[np.where(mf == mf.min())[0]]
    else:
        chosen = comps[np.where(mf <= jf)[0]]
        if len(chosen) == 0:
            chosen = comps[np.where(mf == mf.min())[0]]
    return int(rng.choice(chosen))


def best_pdr_seed(
    instance: Instance,
    scenarios,
    *,
    methods: tuple[PDRName, ...] = ("FIFO", "MOR", "SPT", "MWKR"),
    seed: int = 50,
    obj: str = "mean",
) -> Solution:
    from sfjsp_alns.evaluate import evaluate_solution, objective_value

    best = None
    best_val = np.inf
    for i, method in enumerate(methods):
        sol = pdr_solution(instance, method, seed=seed + i)
        met = evaluate_solution(instance, sol, scenarios, obj=obj)  # type: ignore[arg-type]
        val = objective_value(met, obj)  # type: ignore[arg-type]
        if val < best_val:
            best_val = val
            best = sol
    assert best is not None
    best.meta["best_pdr_obj"] = best_val
    return best
