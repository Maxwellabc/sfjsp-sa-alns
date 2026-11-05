"""Scenario-aware ALNS main loop with SA acceptance (M1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sfjsp_alns.evaluate import evaluate_solution, mean_cmax_fast, objective_value
from sfjsp_alns.operators.destroy import DESTROY_OPS
from sfjsp_alns.operators.repair import REPAIR_OPS
from sfjsp_alns.types import Instance, ScenarioBundle, Solution


@dataclass
class ALNSConfig:
    max_iter: int = 400
    segment: int = 40
    time_limit_s: float | None = 30.0
    q_min: float = 0.08
    q_max: float = 0.22
    t0_ratio: float = 0.03
    cooling: float = 0.995
    reaction: float = 0.25
    obj: Literal["mean", "VaR"] = "mean"
    alpha: float = 0.95
    destroy_names: tuple[str, ...] = ("random_ops",)
    repair_names: tuple[str, ...] = ("spt_biased",)
    seed: int = 50
    k_fast: int = 10
    local_machine_tries: int = 20


@dataclass
class ALNSResult:
    best: Solution
    best_search_obj: float
    history: list[float]
    iters: int


def _roulette(weights: dict[str, float], names: list[str], rng: np.random.Generator) -> str:
    w = np.array([max(weights.get(n, 1.0), 1e-6) for n in names], dtype=np.float64)
    w /= w.sum()
    return str(rng.choice(names, p=w))


def _local_machine_improve(
    instance: Instance,
    sol: Solution,
    scenarios: ScenarioBundle,
    rng: np.random.Generator,
    tries: int,
    idx: np.ndarray,
) -> Solution:
    """Random-op machine reassignment hill climb on scenario subsample."""
    cur = sol.copy()
    cur_sc = mean_cmax_fast(instance, cur, scenarios, idx)
    for _ in range(tries):
        op = int(rng.integers(instance.n_ops))
        comps = instance.compatible_machines(op)
        if len(comps) <= 1:
            continue
        m_new = int(rng.choice(comps))
        if m_new == int(cur.machine_assign[op]):
            continue
        trial = cur.copy()
        trial.machine_assign[op] = m_new
        sc = mean_cmax_fast(instance, trial, scenarios, idx)
        if sc + 1e-9 < cur_sc:
            cur = trial
            cur_sc = sc
    return cur


def run_alns(
    instance: Instance,
    init: Solution,
    scenarios_search: ScenarioBundle,
    scenarios_refine: ScenarioBundle | None = None,
    config: ALNSConfig | None = None,
) -> ALNSResult:
    import time

    cfg = config or ALNSConfig()
    rng = np.random.default_rng(cfg.seed)
    refine = scenarios_refine or scenarios_search

    d_names = list(cfg.destroy_names)
    r_names = list(cfg.repair_names)
    for name in d_names:
        if name not in DESTROY_OPS:
            raise KeyError(name)
    for name in r_names:
        if name not in REPAIR_OPS:
            raise KeyError(name)

    w_d = {n: 1.0 for n in d_names}
    w_r = {n: 1.0 for n in r_names}
    score_d = {n: 0.0 for n in d_names}
    score_r = {n: 0.0 for n in r_names}
    use_d = {n: 0 for n in d_names}
    use_r = {n: 0 for n in r_names}

    k_fast = min(cfg.k_fast, scenarios_search.n_scenarios)
    idx_fast = rng.choice(scenarios_search.n_scenarios, size=k_fast, replace=False)

    def search_obj(sol: Solution) -> float:
        met = evaluate_solution(instance, sol, scenarios_search, obj=cfg.obj, alpha=cfg.alpha)
        return objective_value(met, cfg.obj)

    def refine_obj(sol: Solution) -> float:
        met = evaluate_solution(instance, sol, refine, obj=cfg.obj, alpha=cfg.alpha)
        return objective_value(met, cfg.obj)

    x = init.copy()
    # light local improve on init
    x = _local_machine_improve(
        instance, x, scenarios_search, rng, cfg.local_machine_tries, idx_fast
    )
    fx = search_obj(x)
    x_best = x.copy()
    f_best = refine_obj(x_best)
    history = [f_best]
    T = max(cfg.t0_ratio * fx, 1.0)
    t0 = time.time()
    it_done = 0

    for it in range(1, cfg.max_iter + 1):
        if cfg.time_limit_s is not None and (time.time() - t0) > cfg.time_limit_s:
            break
        it_done = it
        d_name = _roulette(w_d, d_names, rng)
        r_name = _roulette(w_r, r_names, rng)
        q = float(rng.uniform(cfg.q_min, cfg.q_max))
        destroy = DESTROY_OPS[d_name](instance, x, scenarios_search, rng, q)
        y = REPAIR_OPS[r_name](instance, destroy, scenarios_search, rng, cfg.obj)
        y = _local_machine_improve(
            instance, y, scenarios_search, rng, max(5, cfg.local_machine_tries // 2), idx_fast
        )
        fy = search_obj(y)
        delta = fy - fx
        accepted = False
        reward = 0.0
        if delta < 0:
            accepted = True
            reward = 5.0 if fy < fx else 1.0
        elif rng.random() < np.exp(-delta / max(T, 1e-9)):
            accepted = True
            reward = 1.0

        use_d[d_name] += 1
        use_r[r_name] += 1
        if accepted:
            x, fx = y, fy
            score_d[d_name] += reward
            score_r[r_name] += reward
            f_ref = refine_obj(x)
            if f_ref + 1e-9 < f_best:
                x_best, f_best = x.copy(), f_ref
                score_d[d_name] += 10.0
                score_r[r_name] += 10.0
                history.append(f_best)

        T *= cfg.cooling
        if it % cfg.segment == 0:
            for n in d_names:
                if use_d[n] > 0:
                    w_d[n] = (1 - cfg.reaction) * w_d[n] + cfg.reaction * (score_d[n] / use_d[n])
                score_d[n] = 0.0
                use_d[n] = 0
            for n in r_names:
                if use_r[n] > 0:
                    w_r[n] = (1 - cfg.reaction) * w_r[n] + cfg.reaction * (score_r[n] / use_r[n])
                score_r[n] = 0.0
                use_r[n] = 0
            # refresh fast idx
            idx_fast = rng.choice(scenarios_search.n_scenarios, size=k_fast, replace=False)

    x_best.meta = dict(x_best.meta)
    x_best.meta.update({"alns_search_best": f_best, "alns_iters": it_done})
    return ALNSResult(best=x_best, best_search_obj=f_best, history=history, iters=it_done)
