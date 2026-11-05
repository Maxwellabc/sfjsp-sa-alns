"""Deterministic schedule decoder: Solution + op_pt -> Cmax."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sfjsp_alns.instance_utils import first_last_op, op_to_job
from sfjsp_alns.types import Instance, Solution


def decode_cmax(
    instance: Instance,
    solution: Solution,
    op_pt: npt.NDArray[np.int64],
) -> float:
    """
    Earliest-start decoding on machine Gantt with job precedence.

    For each op in ``op_sequence`` order:
      start = max(job_ready[job], machine_ready[m]); end = start + pt[op,m]
    """
    n_ops = instance.n_ops
    n_jobs = instance.n_jobs
    n_m = instance.n_machines
    if solution.machine_assign.shape[0] != n_ops or solution.op_sequence.shape[0] != n_ops:
        raise ValueError("solution size mismatch")

    job_of = op_to_job(instance.job_length)
    job_ready = np.zeros(n_jobs, dtype=np.float64)
    mch_ready = np.zeros(n_m, dtype=np.float64)
    # next expected op per job (precedence check soft-fix by order)
    first, last = first_last_op(instance.job_length)
    next_needed = first.copy()

    for op in solution.op_sequence:
        op = int(op)
        j = int(job_of[op])
        m = int(solution.machine_assign[op])
        if m < 0 or m >= n_m:
            raise ValueError(f"invalid machine {m} for op {op}")
        pt = int(op_pt[op, m])
        if pt <= 0:
            # incompatible under this scenario — treat as large penalty
            return 1e15
        # enforce precedence: if earlier ops of job not yet scheduled, delay via job_ready
        # (sequence should already respect precedence)
        start = job_ready[j] if next_needed[j] == op else max(job_ready[j], mch_ready[m])
        if next_needed[j] != op:
            # sequence violated — still schedule after job_ready
            start = max(job_ready[j], mch_ready[m])
        else:
            start = max(job_ready[j], mch_ready[m])
        end = start + pt
        job_ready[j] = end
        mch_ready[m] = end
        if next_needed[j] == op and op < last[j]:
            next_needed[j] = op + 1
        elif next_needed[j] == op:
            next_needed[j] = op + 1  # job done
    return float(max(mch_ready.max(), job_ready.max()))


def decode_schedule_times(
    instance: Instance,
    solution: Solution,
    op_pt: npt.NDArray[np.int64],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (start, end) arrays of shape (N,)."""
    n_ops = instance.n_ops
    job_of = op_to_job(instance.job_length)
    job_ready = np.zeros(instance.n_jobs, dtype=np.float64)
    mch_ready = np.zeros(instance.n_machines, dtype=np.float64)
    starts = np.zeros(n_ops, dtype=np.float64)
    ends = np.zeros(n_ops, dtype=np.float64)
    for op in solution.op_sequence:
        op = int(op)
        j = int(job_of[op])
        m = int(solution.machine_assign[op])
        pt = int(op_pt[op, m])
        start = max(job_ready[j], mch_ready[m])
        end = start + pt
        starts[op] = start
        ends[op] = end
        job_ready[j] = end
        mch_ready[m] = end
    return starts, ends
