#!/usr/bin/env python3
"""M1 smoke: SA-ALNS vs MOR on 3 SD3 10x5 instances (same protocol as PDR smoke)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/fjsp-combo-b")
sys.path.insert(0, str(ROOT / "src"))

from sfjsp_alns.alns import ALNSConfig, run_alns
from sfjsp_alns.evaluate import evaluate_solution
from sfjsp_alns.init_pdr import best_pdr_seed, pdr_solution
from sfjsp_alns.io.fjsp_loader import load_instance, load_scenario_bundle
from sfjsp_alns.types import ScenarioBundle

INST_DIR = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/smoke_SD3/10x5+mix"
REAL_DIR = (
    ROOT
    / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/stochastic_realizations/random/smoke_SD3/10x5+mix"
)
OUT = ROOT / "results/alns-m1-smoke.md"
SEED = 50
EVAL_START, EVAL_K = 250, 50
SEARCH_START, SEARCH_K = 0, 40  # from pool before eval slice


def load_pool(path: Path) -> np.ndarray:
    """All realization op_pt stacked (K,N,M) using official loader semantics."""
    import sys as _sys

    drl = ROOT / "baselines/NCO-for-Stochastic-FJSP/FJSP-DRL"
    _sys.path.insert(0, str(drl))
    from data_utils import load_data_from_stochastic_realizations_file

    _, op_pts = load_data_from_stochastic_realizations_file(str(path), max_realizations=400)
    return np.stack([np.asarray(p, dtype=np.int64) for p in op_pts], axis=0)


def main() -> None:
    lines: list[str] = []

    def L(s: str = "") -> None:
        print(s)
        lines.append(s)

    L("# SA-ALNS M1 冒烟（vs MOR）")
    L()
    L(f"- 时间：2026-11-05（Asia/Shanghai）")
    L(f"- 协议对齐：`results/stochastic-fjsp-smoke.md`（3 例 SD3 10×5，评测切片 `[250:300]`，K_eval=50，seed={SEED}）")
    L(f"- 搜索场景：realizations `[{SEARCH_START}:{SEARCH_START+SEARCH_K}]`（不与评测切片重叠）")
    L(f"- 方法：best-of-PDR 初解 + D1 `random_ops` + R4 `spt_biased` + 机器局部改进 + SA-ALNS")
    L()

    names = sorted(INST_DIR.glob("*.fjs"))
    rows = []
    t0 = time.time()
    for path in names:
        inst = load_instance(path)
        real_path = REAL_DIR / path.name
        pool = load_pool(real_path)
        if pool.shape[0] < EVAL_START + EVAL_K:
            raise RuntimeError(f"not enough realizations in {real_path}: {pool.shape[0]}")
        eval_sc = ScenarioBundle(
            op_pt_scen=pool[EVAL_START : EVAL_START + EVAL_K],
            role="eval",
            source_slice=(EVAL_START, EVAL_START + EVAL_K),
        )
        search_sc = ScenarioBundle(
            op_pt_scen=pool[SEARCH_START : SEARCH_START + SEARCH_K],
            role="search",
            source_slice=(SEARCH_START, SEARCH_START + SEARCH_K),
        )
        # MOR baseline on eval (our decoder/PDR — may differ slightly from torch env)
        mor = pdr_solution(inst, "MOR", seed=SEED)
        mor_m = evaluate_solution(inst, mor, eval_sc)
        init = best_pdr_seed(inst, search_sc, seed=SEED)
        cfg = ALNSConfig(
            max_iter=350,
            time_limit_s=45.0,
            seed=SEED + hash(path.name) % 1000,
            k_fast=10,
            local_machine_tries=25,
        )
        t1 = time.time()
        res = run_alns(inst, init, search_sc, scenarios_refine=search_sc, config=cfg)
        alns_m = evaluate_solution(inst, res.best, eval_sc)
        elapsed = time.time() - t1
        rows.append(
            {
                "name": path.name,
                "mor_mean": mor_m.mean_cmax,
                "mor_var": mor_m.var_cmax,
                "alns_mean": alns_m.mean_cmax,
                "alns_var": alns_m.var_cmax,
                "iters": res.iters,
                "time": elapsed,
                "gap": (alns_m.mean_cmax - mor_m.mean_cmax) / mor_m.mean_cmax * 100,
            }
        )
        L(
            f"- `{path.name}`: MOR E[Cmax]={mor_m.mean_cmax:.2f} → ALNS {alns_m.mean_cmax:.2f} "
            f"(gap {rows[-1]['gap']:+.2f}%), VaR95 {mor_m.var_cmax:.2f}→{alns_m.var_cmax:.2f}, "
            f"iters={res.iters}, {elapsed:.1f}s"
        )

    mor_avg = float(np.mean([r["mor_mean"] for r in rows]))
    alns_avg = float(np.mean([r["alns_mean"] for r in rows]))
    mor_var_avg = float(np.mean([r["mor_var"] for r in rows]))
    alns_var_avg = float(np.mean([r["alns_var"] for r in rows]))
    # also compare to published smoke MOR 686.97
    smoke_mor = 686.97
    L()
    L("## 总表（评测期望 Cmax / VaR95）")
    L()
    L("| 实例 | MOR E[Cmax] | SA-ALNS E[Cmax] | gap% | MOR VaR95 | ALNS VaR95 | 迭代 | 秒 |")
    L("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        L(
            f"| {r['name']} | {r['mor_mean']:.2f} | {r['alns_mean']:.2f} | {r['gap']:+.2f} | "
            f"{r['mor_var']:.2f} | {r['alns_var']:.2f} | {r['iters']} | {r['time']:.1f} |"
        )
    L(
        f"| **均值** | **{mor_avg:.2f}** | **{alns_avg:.2f}** | "
        f"**{(alns_avg - mor_avg) / mor_avg * 100:+.2f}** | **{mor_var_avg:.2f}** | **{alns_var_avg:.2f}** | — | — |"
    )
    L()
    L("## 与 PDR 冒烟报告对齐")
    L()
    L(f"| 参照 | 期望 Cmax 均值 |")
    L(f"|---|---:|")
    L(f"| 官方 env PDR 冒烟 MOR（`stochastic-fjsp-smoke.md`） | {smoke_mor:.2f} |")
    L(f"| 本包 decoder 上 MOR | {mor_avg:.2f} |")
    L(f"| 本包 SA-ALNS M1 | {alns_avg:.2f} |")
    beat_ours = alns_avg < mor_avg
    beat_smoke = alns_avg < smoke_mor
    L()
    L("## 结论")
    L()
    L(
        f"- vs 本包 MOR：{'**PASS**' if beat_ours else 'FAIL'}（{alns_avg:.2f} {'<' if beat_ours else '>='} {mor_avg:.2f}）"
    )
    L(
        f"- vs 冒烟 MOR 686.97：{'**PASS**' if beat_smoke else 'FAIL'}（{alns_avg:.2f} {'<' if beat_smoke else '>='} {smoke_mor:.2f}）"
    )
    L(f"- 总耗时 {time.time() - t0:.1f}s")
    L()
    L("```bash")
    L("cd /workspace/fjsp-combo-b && source .venv/bin/activate")
    L("PYTHONPATH=src python scripts/alns_m1_smoke.py")
    L("```")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    if not (beat_ours and beat_smoke):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
