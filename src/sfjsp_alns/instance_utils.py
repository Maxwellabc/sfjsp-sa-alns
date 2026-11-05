"""Helpers for job ↔ operation indexing."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sfjsp_alns.types import Instance


def first_last_op(job_length: npt.NDArray[np.int64]) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros(len(job_length), dtype=np.int64)
    first[1:] = np.cumsum(job_length[:-1])
    last = first + job_length - 1
    return first, last


def op_to_job(job_length: npt.NDArray[np.int64]) -> np.ndarray:
    return np.repeat(np.arange(len(job_length), dtype=np.int64), job_length)


def assert_precedence(instance: Instance, op_sequence: npt.NDArray[np.int64]) -> None:
    first, last = first_last_op(instance.job_length)
    pos = np.empty(instance.n_ops, dtype=np.int64)
    pos[op_sequence] = np.arange(instance.n_ops)
    for j in range(instance.n_jobs):
        for op in range(first[j], last[j]):
            if pos[op] > pos[op + 1]:
                raise ValueError(f"precedence violated job={j} ops {op}->{op+1}")
