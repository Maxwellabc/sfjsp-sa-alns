from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Instance:
    """Deterministic FJSP instance aligned with Smit FJSP-DRL .fjs loading."""

    name: str
    job_length: npt.NDArray[np.int64]  # (J,)
    op_pt_nom: npt.NDArray[np.int64]  # (N, M); 0 = incompatible
    path: str | None = None

    @property
    def n_jobs(self) -> int:
        return int(self.job_length.shape[0])

    @property
    def n_ops(self) -> int:
        return int(self.op_pt_nom.shape[0])

    @property
    def n_machines(self) -> int:
        return int(self.op_pt_nom.shape[1])

    def compatible_machines(self, op: int) -> npt.NDArray[np.int64]:
        row = self.op_pt_nom[op]
        return np.flatnonzero(row > 0).astype(np.int64)


@dataclass
class ScenarioBundle:
    """K processing-time scenarios; role separates search vs eval pools."""

    op_pt_scen: npt.NDArray[np.int64]  # (K, N, M)
    role: Literal["search", "eval", "pool"] = "pool"
    source_slice: tuple[int, int] | None = None

    @property
    def n_scenarios(self) -> int:
        return int(self.op_pt_scen.shape[0])


@dataclass
class Solution:
    """Two-layer encoding: machine assignment + global operation order."""

    machine_assign: npt.NDArray[np.int64]  # (N,)
    op_sequence: npt.NDArray[np.int64]  # (N,) permutation-like with precedence
    meta: dict = field(default_factory=dict)

    def copy(self) -> "Solution":
        return Solution(
            machine_assign=self.machine_assign.copy(),
            op_sequence=self.op_sequence.copy(),
            meta=dict(self.meta),
        )


@dataclass
class Metrics:
    mean_cmax: float
    var_cmax: float
    alpha: float = 0.95
    n_scenarios: int = 0
    decode_count: int = 0
