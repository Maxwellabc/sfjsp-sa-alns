# Draft: Experiments

**Working title:** Scenario-Aware Adaptive Large Neighborhood Search for Stochastic Flexible Job Shop Scheduling  
**Venue target:** IJCAI 2027 (retargetable to AAAI)  
**Status:** English draft for the Experiments section. **All tables below use numbers measured in this repository on 2026-11-05 (Asia/Shanghai).** Claims are limited to methods that were actually run.  
**Raw artifacts:** `results/p0-sd3-10x5.md`, `results/p0-sd3-10x5-raw.csv`, `results/p0-sd3-10x5-summary.json`, `results/p0-ablation-destroy.csv`, `results/p0-cp-stoch.md`, `results/p0-baselines-compare.md`  
**Date:** 2026-11-05 (Asia/Shanghai)

---

## 中文节注

- 主实验：SD3 `10x5+mix` 前 **20** 例 × 3 种子；K_eval=**100**（非论文级 1000）；搜索 `[0:50]` / 评测 `[250:350]`。
- SA-ALNS 宏平均 E[Cmax]=**597.16**，优于 MOR **654.31**（约 -8.7%）；20/20 实例、60/60 种子级均优于 MOR。
- CP-stoch（短时限）：官方 mean CP-SAT，**60s**，K_cp=10；held-out 宏平均 E[Cmax]=**618.92**、VaR95=**722.35**；SA-ALNS 相对其约 **-3.5%**，实例级 **18/20** 更优。**非**论文 3600s 设定。
- SPM-DAN：**N/A**（公开仓无 FJSP 预训练权重；短训不公平且缺 gym/torch_geometric/einops）。
- 场景库为本轮按官方 lognormal random-CoV 配方新生成（450 条/例），与早期 `smoke_SD3` **不可直接数值对齐**。

---

## 5. Experiments

### 5.1 Experimental setup

**Instances.** We evaluate on the first 20 instances of the SD3 `10x5+mix` family released with Smit et al. [2025] (10 jobs × 5 machines). Full SD3 contains 100 instances in this size class; expanding to all 100 and to larger sizes (e.g., 20×5, 15×10) is left for the camera-ready / follow-up pass.

**Stochastic realizations.** Processing-time scenarios are generated with the official *random* lognormal coefficient-of-variation recipe (`create_stochastic_realizations_random_variances`, `seed_datagen=200`) and written under `stochastic_realizations/random/SD3/10x5+mix/`. Each file contains 450 realizations. This bank is **protocol-compatible** with Smit et al. but is **not** bitwise identical to the three-instance `smoke_SD3` bank used in earlier smoke tests; therefore smoke numbers must not be mixed into the tables below.

**Search / evaluation split.** Search uses realizations `[0:50]`. Reporting uses the held-out slice `[250:350]` (**K_eval=100**). We do **not** tune on the evaluation slice. The Smit et al. paper often reports with K_eval=1000; that setting requires a larger realization bank and is listed as remaining work.

**Methods compared (executed).**
- Priority dispatching rules (PDRs): FIFO, MOR, SPT, MWKR, implemented in-package on the same decoder.
- **SA-ALNS:** best-of-PDR initialization on the search set; destroy portfolio `{random_ops, critical_path, high_var_ops, machine_load}`; repair `spt_biased`; roulette weights with simulated-annealing acceptance; up to 300 iterations or 40s per run.

**Additional baseline executed (short budget).**
- **CP-stoch (mean):** official OR-Tools model in `Job_Shop_Scheduling_Benchmark_Environments_and_Instances` (`FJSPmodel_stoch`, objective `mean`). Search on the first **K_cp=10** realizations; wall-clock limit **60s** per instance (honest short budget; Smit et al. report up to **3600s**). The CP assignment and start-order are converted to our two-layer encoding and **re-evaluated** on `[250:350]` with the same decoder as PDRs/ALNS. Artifacts: `results/p0-cp-stoch.*`, `results/p0-baselines-compare.*`.

**Methods still N/A.**
- **SPM-DAN / DAN-stoch:** public NCO-for-Stochastic-FJSP provides **no** pretrained FJSP checkpoints (confirmed against GitHub/releases); a short CPU retrain would be unfair vs the paper and currently lacks `gym` / `torch_geometric` / `einops` in the P0 venv. We do not invent scores.

**Seeds and aggregation.** Seeds `{50,51,52}` for PDRs/ALNS. CP-stoch is deterministic given the scenario bank (single run per instance). For each (instance, method) we average E[$C_{\max}$] and VaR$_{95}$ over seeds where applicable, then macro-average over the 20 instances. ALNS suite wall-clock ≈632s; CP-stoch suite ≈1215s (≈60s/instance).

### 5.2 Main results (P0)

Table 1 reports macro-averaged expected makespan and VaR$_{95}$ on the evaluation slice.

**Table 1.** Macro averages on 20× SD3 `10x5+mix`, K_eval=100, 3 seeds. Gap is relative to MOR.

| Method | E[$C_{\max}$] | VaR$_{95}$ | vs MOR (%) |
|--------|----------------:|-------------:|-----------:|
| FIFO | 661.11 | 770.34 | +1.04 |
| MOR | 654.31 | 762.64 | +0.00 |
| SPT | 823.84 | 943.42 | +25.91 |
| MWKR | 654.18 | 763.53 | -0.02 |
| **SA-ALNS** | **597.16** | **683.98** | **-8.73** |
| CP-stoch (60s, K=10) | 618.92 | 722.35 | -5.41 |
| SPM-DAN | N/A | N/A | — |

SA-ALNS improves mean expected makespan by **57.15** absolute points versus MOR (-8.73%). It also reduces macro VaR$_{95}$ from 762.64 to 683.98. Among PDRs, MWKR (654.18) is essentially tied with MOR; SPT is substantially worse. Under the **short** CP budget, CP-stoch improves on MOR by 5.41% (618.92) but remains **3.51%** above SA-ALNS (597.16). **These CP numbers are not comparable to Smit et al.'s 3600s runs.**

**Win counts.** On seed-mean E[$C_{\max}$], SA-ALNS beats MOR on **20/20** instances. At the (instance, seed) grain, ALNS is strictly better than the same-seed MOR on **60/60** runs. Versus CP-stoch-60s, SA-ALNS seed-means win on **18/20** instances (losses: `10x5+mix_004`, `10x5+mix_012`, gaps +6.79 / +1.84).

**Table 2.** Per-instance seed-mean E[$C_{\max}$] (excerpt of full table in `results/p0-sd3-10x5.md`).

| Instance | MOR | SA-ALNS | Δ |
|----------|----:|--------:|--:|
| 10x5+mix_001.fjs | 741.66 | 654.59 | -87.07 |
| 10x5+mix_002.fjs | 679.74 | 641.65 | -38.08 |
| 10x5+mix_003.fjs | 665.37 | 616.39 | -48.97 |
| 10x5+mix_004.fjs | 635.13 | 571.57 | -63.56 |
| 10x5+mix_005.fjs | 583.81 | 551.30 | -32.51 |
| 10x5+mix_006.fjs | 612.00 | 562.26 | -49.74 |
| 10x5+mix_007.fjs | 558.59 | 517.34 | -41.24 |
| 10x5+mix_008.fjs | 843.46 | 710.47 | -132.99 |
| 10x5+mix_009.fjs | 717.59 | 661.08 | -56.51 |
| 10x5+mix_010.fjs | 649.39 | 592.84 | -56.55 |
| 10x5+mix_011.fjs | 575.36 | 524.18 | -51.19 |
| 10x5+mix_012.fjs | 635.18 | 609.64 | -25.55 |
| 10x5+mix_013.fjs | 676.73 | 638.25 | -38.49 |
| 10x5+mix_014.fjs | 697.12 | 636.77 | -60.36 |
| 10x5+mix_015.fjs | 643.11 | 562.24 | -80.87 |
| 10x5+mix_016.fjs | 650.81 | 596.56 | -54.26 |
| 10x5+mix_017.fjs | 639.57 | 583.44 | -56.14 |
| 10x5+mix_018.fjs | 641.25 | 579.38 | -61.87 |
| 10x5+mix_019.fjs | 636.23 | 583.36 | -52.87 |
| 10x5+mix_020.fjs | 604.09 | 549.96 | -54.12 |

### 5.3 Ablation: scenario-aware destroy operators

On the first five instances with seed 50 and identical budgets, we compare a D1-only destroy portfolio against the full D1+D3–D5 set (same repair and SA loop).

**Table 3.** Mean E[$C_{\max}$] over 5 instances (seed 50).

| Variant | Destroy set | Mean E[$C_{\max}$] |
|---------|-------------|----------------------:|
| D1 only | `random_ops` | 610.48 |
| Full (P0 default) | `random_ops` + `critical_path` + `high_var_ops` + `machine_load` | 604.19 |

The scenario-aware portfolio yields a modest average improvement of **6.29** expected makespan points on this small probe (-1.03% vs D1-only). Gains are not uniform across instances (see `results/p0-ablation-destroy.csv`); a larger multi-seed ablation remains necessary before strong claims about individual operators.

### 5.4 Discussion and limitations

1. **Scope.** Results cover 20/100 SD3 `10x5` instances and K_eval=100, not the full paper protocol (100 instances, K_eval=1000, larger sizes).
2. **Baselines.** SPM-DAN remains **N/A** (no public FJSP weights). CP-stoch is run only under a **60s / K_cp=10 mean** short budget with held-out re-decode; we **do not** claim wins over Smit et al.'s long-budget CP or trained SPM-DAN.
3. **Decoder / PDR stack.** All PDR, ALNS, and CP re-eval numbers share our standalone decoder. Absolute values are internally consistent but may differ from the official torch-env PDR smoke numbers on `smoke_SD3`.
4. **Compute.** ALNS uses a short budget (~10s / 300 iterations); CP uses 60s. Longer budgets and time–quality curves (including paper-matched CP) are future work.
5. **LLM.** No LLM operator selection was used; the present results are the non-LLM strong baseline.

### 5.5 Remaining P0/P1 checklist

| Item | Status |
|------|--------|
| 20× SD3 10×5, 3 seeds, vs FIFO/MOR/SPT/MWKR | **Done** (this draft) |
| D3–D5 implemented + small ablation | **Done** (probe) |
| Full 100× 10×5 | Pending |
| K_eval=1000 | Pending (need ≥1250 realizations/file) |
| SPM-DAN eval or self-train | **N/A blocked** (no weights; fair train pending) |
| CP-SAT / CP-stoch short (60s, K=10, mean) | **Done** (`results/p0-cp-stoch.*`) |
| CP-stoch paper budget (≤3600s, more scenarios / VaR) | Pending |
| Larger SD3 sizes | Pending |
| Multi-seed full destroy ablation | Pending |

### 5.6 Reproducibility

```bash
cd /workspace/fjsp-combo-b && source .venv/bin/activate
SFJSP_REAL_LIMIT=20 PYTHONPATH=baselines/NCO-for-Stochastic-FJSP/FJSP-DRL \
  python scripts/generate_sd3_10x5_realizations.py
PYTHONPATH=src python scripts/alns_p0_sd3_10x5.py
PYTHONPATH=src python scripts/run_cp_stoch_p0.py
```

---

*End of Experiments draft. Numbers only from measured CSV/JSON artifacts.*
