"""Held-out K_eval evaluation for SA-ALNS.

Search and reporting use disjoint scenario slices. The paper protocol is
search ``[0:50]`` and eval ``[250:250+K_eval]`` with K_eval=1000. This module
only scores solutions; it does not tune on the eval slice.

Synthetic fixtures exercise the same ``run_alns`` / ``evaluate_solution`` path
without the external SD3 bank. Their JSON is marked
``comparable_to_paper_protocol=false``.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from sfjsp_alns.alns import ALNSConfig, run_alns
from sfjsp_alns.evaluate import evaluate_solution
from sfjsp_alns.init_pdr import best_pdr_seed, pdr_solution
from sfjsp_alns.scenarios import lognormal_random_cov
from sfjsp_alns.types import Instance, ScenarioBundle

PDRS: tuple[str, ...] = ("FIFO", "MOR", "SPT", "MWKR")
DESTROY: tuple[str, ...] = ("random_ops", "critical_path", "high_var_ops", "machine_load")
REPAIR: tuple[str, ...] = ("spt_biased",)

SPMDAN_STATUS = (
    "N/A — pretrained SPM-DAN/DAN-stoch weights are not in the public "
    "NCO-for-Stochastic-FJSP checkout; this harness does not invent a score."
)


@dataclass(frozen=True)
class KEvalSpec:
    k_eval: int = 1000
    eval_start: int = 250
    k_search: int = 50
    search_start: int = 0
    seeds: tuple[int, ...] = (50,)
    max_iter: int = 300
    time_limit_s: float = 40.0
    alpha: float = 0.95
    kind: str = "external_sd3"

    @property
    def search_stop(self) -> int:
        return self.search_start + self.k_search

    @property
    def eval_stop(self) -> int:
        return self.eval_start + self.k_eval

    @property
    def pool_length(self) -> int:
        return max(self.search_stop, self.eval_stop)


def assert_slices_disjoint(spec: KEvalSpec) -> None:
    if spec.k_eval < 1 or spec.k_search < 1:
        raise ValueError("k_eval and k_search must be >= 1")
    if spec.search_start < 0 or spec.eval_start < 0:
        raise ValueError("slice starts must be >= 0")
    if spec.time_limit_s <= 0 or spec.max_iter < 1:
        raise ValueError("time_limit and max_iter must be positive")
    s0, s1 = spec.search_start, spec.search_stop
    e0, e1 = spec.eval_start, spec.eval_stop
    if not (s1 <= e0 or e1 <= s0):
        raise ValueError(
            f"search [{s0}:{s1}] overlaps eval [{e0}:{e1}]. "
            "Held-out K_eval must not be used for search or operator selection."
        )


def stable_seed(base: int, name: str) -> int:
    """Process-stable mix. Do not use builtin hash(); it is salted per process."""
    return int(base) + (zlib.crc32(name.encode("utf-8")) % 997)


def synthetic_pool(instance: Instance, n_scenarios: int, seed: int) -> np.ndarray:
    """In-process lognormal random-CoV pool. Not an external SD3 bank."""
    rng = np.random.default_rng(stable_seed(seed, instance.name))
    return lognormal_random_cov(instance.op_pt_nom, n_scenarios, rng)


def _bundles(pool: np.ndarray, spec: KEvalSpec) -> tuple[ScenarioBundle, ScenarioBundle]:
    if pool.shape[0] < spec.pool_length:
        raise ValueError(
            f"realization pool has {pool.shape[0]} scenarios; "
            f"need >= {spec.pool_length} for search [{spec.search_start}:{spec.search_stop}] "
            f"and eval [{spec.eval_start}:{spec.eval_stop}]. "
            "External banks: regenerate with NUM_SAMPLES >= eval_start + k_eval "
            "(1250 when K_eval=1000 and eval_start=250)."
        )
    search = ScenarioBundle(
        op_pt_scen=pool[spec.search_start : spec.search_stop],
        role="search",
        source_slice=(spec.search_start, spec.search_stop),
    )
    held = ScenarioBundle(
        op_pt_scen=pool[spec.eval_start : spec.eval_stop],
        role="eval",
        source_slice=(spec.eval_start, spec.eval_stop),
    )
    return search, held


def evaluate_instance(instance: Instance, pool: np.ndarray, spec: KEvalSpec) -> list[dict]:
    """Run PDRs and SA-ALNS. Returns one row per (seed, method)."""
    assert_slices_disjoint(spec)
    search, held = _bundles(pool, spec)
    rows: list[dict] = []
    for seed in spec.seeds:
        for pdr in PDRS:
            sol = pdr_solution(instance, pdr, seed=seed)  # type: ignore[arg-type]
            metrics = evaluate_solution(instance, sol, held, alpha=spec.alpha)
            rows.append(
                {
                    "instance": instance.name,
                    "method": pdr,
                    "seed": int(seed),
                    "mean_cmax": metrics.mean_cmax,
                    "var95": metrics.var_cmax,
                    "iters": 0,
                    "seconds": 0.0,
                    "n_eval": metrics.n_scenarios,
                }
            )
        init = best_pdr_seed(instance, search, seed=seed)
        cfg = ALNSConfig(
            max_iter=spec.max_iter,
            time_limit_s=spec.time_limit_s,
            seed=stable_seed(seed, instance.name),
            k_fast=min(10, spec.k_search),
            local_machine_tries=20,
            destroy_names=DESTROY,
            repair_names=REPAIR,
            alpha=spec.alpha,
        )
        t1 = time.time()
        result = run_alns(instance, init, search, scenarios_refine=search, config=cfg)
        elapsed = time.time() - t1
        metrics = evaluate_solution(instance, result.best, held, alpha=spec.alpha)
        rows.append(
            {
                "instance": instance.name,
                "method": "SA-ALNS",
                "seed": int(seed),
                "mean_cmax": metrics.mean_cmax,
                "var95": metrics.var_cmax,
                "iters": int(result.iters),
                "seconds": float(elapsed),
                "n_eval": metrics.n_scenarios,
            }
        )
    return rows


def aggregate(rows: list[dict], spec: KEvalSpec, *, n_instances: int) -> dict:
    methods = list(PDRS) + ["SA-ALNS"]
    per_inst: dict[str, dict] = {}
    macro_e: dict[str, list[float]] = {m: [] for m in methods}
    macro_v: dict[str, list[float]] = {m: [] for m in methods}
    for name in sorted({r["instance"] for r in rows}):
        per_inst[name] = {}
        for method in methods:
            sub = [r for r in rows if r["instance"] == name and r["method"] == method]
            mean_e = float(np.mean([r["mean_cmax"] for r in sub]))
            mean_v = float(np.mean([r["var95"] for r in sub]))
            per_inst[name][method] = {
                "mean_cmax": mean_e,
                "std_cmax": float(np.std([r["mean_cmax"] for r in sub], ddof=0)),
                "var95": mean_v,
                "seconds": float(np.mean([r["seconds"] for r in sub])),
                "iters": float(np.mean([r["iters"] for r in sub])),
            }
            macro_e[method].append(mean_e)
            macro_v[method].append(mean_v)
    paper = (
        spec.kind == "external_sd3"
        and spec.k_eval == 1000
        and spec.eval_start == 250
        and spec.search_start == 0
        and spec.k_search == 50
    )
    return {
        "protocol": {
            "kind": spec.kind,
            "k_eval": spec.k_eval,
            "eval_slice": [spec.eval_start, spec.eval_stop],
            "search_slice": [spec.search_start, spec.search_stop],
            "seeds": list(spec.seeds),
            "max_iter": spec.max_iter,
            "time_limit_s": spec.time_limit_s,
            "destroy": list(DESTROY),
            "repair": list(REPAIR),
            "alpha": spec.alpha,
            "comparable_to_paper_protocol": paper,
            "note": (
                "Eval slice is held out from search. "
                "Synthetic kind uses shipped tiny fixtures and an in-process "
                "lognormal random-CoV draw; it is a wiring check, not an SD3 table. "
                "External kind reads --data-root and does not resample the bank."
            ),
        },
        "n_instances": n_instances,
        "macro_mean_cmax": {m: float(np.mean(macro_e[m])) for m in methods},
        "macro_mean_var95": {m: float(np.mean(macro_v[m])) for m in methods},
        "per_instance": per_inst,
        "rows": rows,
        "spmdan": SPMDAN_STATUS,
    }
