# Paper Outline — Scenario-Aware ALNS for Stochastic FJSP

**Primary target:** IJCAI 2027 Main Track（摘要约 2027-01-04 / 全文约 2027-01-11 AoE）  
**Retarget note:** Same manuscript can retarget **AAAI 2028** (or a later AAAI) if IJCAI timing slips; narrative already contrasts AAAI 2025 Smit et al. NCO baseline.  
**Status of evidence (honest):** Only **M1 smoke** on 3× SD3 `10×5+mix` instances, K_eval=50. **Not** submission-ready experiments.  
**Local artifacts:** `results/alns-m1-smoke.md`, `docs/scenario-aware-alns-design.md`, `src/sfjsp_alns/`  
**Date:** 2026-11-05（Asia/Shanghai）

---

## 0. Title options / 题目候选

| # | English | 中文 |
|---|---------|------|
| A | Scenario-Aware Adaptive Large Neighborhood Search for Stochastic Flexible Job Shop Scheduling | 面向随机柔性作业车间调度的场景感知自适应大邻域搜索 |
| B | Explicit Scenario Sampling Beats Pure Neural Policies? An ALNS Study on Stochastic FJSP | 显式场景采样能否挑战纯神经策略？随机 FJSP 上的 ALNS 研究 |
| C | Search over Scenarios: Metaheuristics for Expectation and VaR Objectives in Stochastic FJSP | 场景上的搜索：随机 FJSP 期望与 VaR 目标的元启发式 |
| D *(LLM variant, only if M4 done)* | LLM-Guided Operator Selection for Scenario-Aware ALNS on Stochastic FJSP | 面向随机 FJSP 场景感知 ALNS 的 LLM 算子选择 |

**Recommendation:** Title **A** for main claim; keep D as optional subtitle / contribution bullet only after ablation.

---

## 1. Abstract bullets / 摘要要点（中英对照）

### Problem / 问题
- EN: Stochastic FJSP with uncertain processing times; optimize expected makespan and/or VaR₉₅ under sampled scenarios.  
- 中：加工时间不确定的 FJSP；在场景样本上优化期望完工时间与/或 VaR₉₅。

### Gap / 缺口
- EN: AAAI 2025 Smit et al. advance NCO via scenario-attention (SPM-DAN) but provide neural policies that need trained weights; classical PDRs remain weak; searchable, reproducible metaheuristics on the **same** stochastic protocol are under-explored.  
- 中：AAAI 2025 用场景注意力 NCO（SPM-DAN）推进学习方法，但依赖训练权重；经典派工规则偏弱；在**同一**随机协议上可复现、可解释的元启发式仍不足。

### Method / 方法
- EN: Two-layer encoding (machine assignment + precedence-feasible op sequence); scenario-aware ALNS with search/eval scenario split; destroy/repair including planned critical-path / high-variance / load operators; optional LLM operator selection (phase-2).  
- 中：两层编码（机器分配 + 工序序）；搜索/评测场景分离的场景感知 ALNS；含关键路径/高方差/载荷等 destroy（规划中）与 SPT 偏置 repair；可选 LLM 算子选择（阶段 2）。

### Results (current vs planned) / 结果（现状 vs 计划）
- EN **current (smoke only):** On 3 SD3 10×5 instances, K_eval=50, SA-ALNS mean E[Cmax]=**640.59** vs package MOR 710.80 (−9.9%) and official-env smoke MOR **686.97**. VaR₉₅ mean 738.55 vs 820.65.  
- 中 **现状：** 仅 3 例冒烟；期望 Cmax 均值 640.59，优于本包 MOR 与官方 env 冒烟 MOR。  
- EN **planned for paper:** Full SD3/SD1 sizes; K_eval=1000; CP-SAT / CP-stoch; SPM-DAN if weights available or self-trained; ablations D3–D5; multi-seed; wall-clock budgets.  
- 中 **投稿前必补：** 全量实例、K_eval=1000、CP/CP-stoch、SPM-DAN（有权重或自训）、算子消融、多种子、时限协议。

### Takeaway / 收束
- EN: Explicit scenario search yields competitive, weight-free schedules and a clear ablation path to LLM-aided control—without replacing the strong non-LLM baseline.  
- 中：显式场景搜索给出无权重、可消融的强基线；LLM 只作增量，不作唯一卖点。

---

## 2. Introduction — contribution claims / 引言贡献声明（必须诚实）

### What we **can** claim now / 现可写的贡献（草稿级）
1. **Protocol-aligned metaheuristic framing** for stochastic FJSP matching Smit’s instance families (SD3/SD1) and E[Cmax]/VaR reporting.  
2. **Implementable SA-ALNS** with search–eval scenario separation to reduce scenario overfitting.  
3. **Preliminary evidence** that ALNS improves over MOR under a fixed small smoke protocol.

### What we **must not** claim yet / 尚未可宣称
1. Outperforming **SPM-DAN / DAN-stoch / CP-stoch** on paper tables — weights missing; CP not run; K_eval≠1000.  
2. Statistical significance on large instance suites — only 3 instances, single seed family.  
3. LLM improves scheduling — M4 not implemented.  
4. Equivalence of our MOR numbers to official torch-env MOR — decoder/PDR details differ (710.80 vs 686.97).

### Differentiation from group NS4S (IJCAI 2025) / 与组内 NS4S 边界
- NS4S: deterministic JSP/FJSP + LLM neighborhood search.  
- This work: **stochastic** objectives + scenario-aware operators; LLM optional and secondary.

---

## 3. Related work outline / 相关工作结构

| Block | EN focus | 中文要点 | Key refs |
|-------|----------|----------|----------|
| R1 Stochastic / robust shop scheduling | Sample average approximation, VaR/CVaR in scheduling | 随机/鲁棒车间；SAA；VaR | classic OR + recent SFJSP |
| R2 NCO for shop scheduling | DAN, L2D, FJSP-DRL; Smit AAAI 2025 SPM | 神经组合优化车间调度 | Wang/Lei DAN; Zhang L2D; **Smit AAAI 2025** |
| R3 LNS/ALNS for scheduling | Destroy–repair, adaptive weights | 调度中的 LNS/ALNS | Ropke & Pisinger lineage; shop-floor LNS |
| R4 LLM for CO (optional §) | Prompted heuristics, NS4S | LLM 辅助组合优化 | NS4S IJCAI 2025; MEoH-style |

Keep R4 short unless M4 ships.

---

## 4. Problem formulation / 问题形式（论文 Method 前半）

### EN
- Instance: jobs, machines, compatible processing times \(p_{ijm}\).  
- Uncertainty: scenarios \(s\in\mathcal{S}\) with realized \(p^{s}\).  
- Solution \(x\): machine assignment + operation sequence (precedence-feasible).  
- Objectives: \(\hat{\mathbb{E}}[C_{\max}(x)]\), \(\widehat{\mathrm{VaR}}_{\alpha}(x)\) on a **held-out** eval scenario set.

### 中
- 解 \(x\)：机器分配 + 满足工件内先后的工序序列；在评测场景集上报告期望 Cmax 与 VaRα。  
- **搜索场景与评测场景分离**（防过拟合）。

---

## 5. Method section outline / 方法章节提纲

### 5.1 Encoding & decoder / 编码与解码
- Two-layer solution; earliest-start Gantt decoder (deterministic).  
- Align `.fjs` / realization file format with Smit open-source loaders.

### 5.2 Initialization / 初始解
- PDR seeds: FIFO, MOR, SPT, MWKR; best-of on search scenarios.

### 5.3 Destroy / Repair (paper table)
| ID | Status for paper | Note |
|----|------------------|------|
| D1 random_ops | **Implemented (M1)** | Clear machines; keep sequence |
| D2 random_seq | Planned | |
| D3 critical_path | Planned (M2) | Scenario-aware |
| D4 high_var_ops | Planned (M2) | Scenario-aware |
| D5 machine_load | Planned (M2) | Scenario-aware |
| D6 job_block | Planned | |
| R4 spt_biased | **Implemented (M1)** | + light relocate |
| R1/R3 greedy/regret | Planned | |

### 5.4 ALNS loop / 主循环
- Roulette adaptive weights; SA acceptance; segment updates; optional local machine hill-climb; optional ILS outer loop.

### 5.5 Optional LLM operator selection / 可选 LLM
- Features → choose (D,R) or \(q\); fallback roulette; **ablation required**.

### 5.6 Complexity / budget / 复杂度
- Decode count as primary budget; wall-clock secondary; fair vs CP 1h / NCO inference time.

---

## 6. Experiments plan / 实验设计（投稿前必完成）

### 6.1 Current smoke (do **not** present as main table alone)
| Method | Mean E[Cmax] | Mean VaR95 |
|--------|-------------:|-----------:|
| Official-env MOR (PDR smoke) | 686.97 | 793.85 |
| Ours MOR | 710.80 | 820.65 |
| **SA-ALNS M1** | **640.59** | **738.55** |

Protocol: 3 instances `10x5+mix_001–003`; K_eval=50 slice `[250:300]`; search `[0:40]`; seed=50; ~350 ALNS iters / ~12s per instance.

### 6.2 Missing experiments checklist / 缺失实验清单

| # | Experiment | Why needed | Priority |
|---|------------|------------|----------|
| E1 | Full SD3 `10x5+mix` (100 inst.) + K_eval=1000 | Match Smit reporting scale | P0 |
| E2 | SD3 larger sizes 20×5, 15×10, … | Scaling claim | P0 |
| E3 | SD1 + mk/rdata noise | Cross-distribution | P1 |
| E4 | Multi-seed (≥5) mean±std | Statistical honesty | P0 |
| E5 | CP-SAT (median) + CP-stoch | Paper baselines in Smit | P0 |
| E6 | SPM-DAN / DAN-stoch | Core contrast; N/A if no weights—then self-train or cite gap | P0 |
| E7 | Ablation: w/o D3–D5; w/o SA; w/o search–eval split | Method justification | P0 |
| E8 | Time–quality curves / decode budgets | Fairness vs NCO | P1 |
| E9 | LLM-on vs LLM-off | Only if M4 | P2 |
| E10 | Align MOR with official env (or always report both) | Credibility | P1 |

### 6.3 Metrics / 指标
- E[Cmax], VaR₉₅, gap% vs MOR and vs best known; wall-clock; #decodes.

### 6.4 Threats to validity / 效度威胁（单独一小节）
1. Smoke-scale optimism; scenario pool generation variance.  
2. Decoder/PDR mismatch vs Smit env → always dual-report.  
3. Missing neural baselines until weights/training.  
4. Adaptive ALNS sensitivity to seeds/hyperparameters.  
5. Possible search-scenario leakage if protocol violated.

---

## 7. Paper skeleton (section order) / 正文目录

1. Introduction  
2. Related Work  
3. Problem Definition  
4. Scenario-Aware ALNS  
   4.1 Encoding & Evaluation Protocol  
   4.2 Operators  
   4.3 Adaptive Search  
   4.4 *(Optional)* LLM-Guided Selection  
5. Experiments  
6. Conclusion & Limitations  
7. Ethics / Reproducibility checklist (code, seeds, scenario slices)

**Page budget:** IJCAI short paper vs full—prefer **full** if E1–E7 land; else workshop/AAAI retarget.

---

## 8. Timeline to Jan 2027 IJCAI deadline / 倒排（Asia/Shanghai）

| Window | Milestone | Deliverable |
|--------|-----------|-------------|
| **2026-11-05 → 11-16** | M2 operators + ablation harness; start full 10×5 | Ablation CSV v0 |
| **11-17 → 11-30** | E1–E2 scaling; CP-stoch runs; attempt SPM-DAN | Baseline table draft |
| **12-01 → 12-14** | E4 multi-seed; polish protocol doc; skip ICAPS | Stable numbers |
| **12-15 → 12-28** | E6 neural baseline decision; figures; related work | Draft v1 |
| **12-29 → 01-03** | Abstract + intro freeze | IJCAI abstract |
| **01-04 → 01-11** | Full paper polish, checklist, code freeze | **IJCAI full submit** |
| **Fallback** | If E5–E6 incomplete by 12-20 | Retarget AAAI 2028 / strengthen CSCWD sister paper |

**Parallel:** CSCWD FJSP-AGV as application sister (deterministic AGV), not blocking IJCAI SFJSP.

---

## 9. Writing todos (immediate) / 立刻写作任务

1. Freeze **contribution paragraph** using §2 wording (no SPM-DAN win claim).  
2. One **protocol figure**: search vs eval scenario slices.  
3. One **method figure**: encoding + ALNS loop.  
4. Placeholder **main result table** with “TBD” rows for CP / SPM-DAN.  
5. Reproducibility appendix: paths to `src/sfjsp_alns`, seeds, slices.

---

## 10. Key citations (verified) / 核实引用

- Smit, Wu, Troubil, Zhang, Nuijten. Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems. AAAI 2025. DOI `10.1609/aaai.v39i25.34870`; arXiv `2412.14052`; code `ai-for-decision-making-tue/NCO-for-Stochastic-FJSP`.  
- NS4S (group). IJCAI 2025 — deterministic LLM neighborhood search; boundary only.  
- Classic PDRs / OR-Tools CP-SAT as in Smit baselines.

---

*This outline is a writing scaffold, not a results claim beyond `alns-m1-smoke.md`.*
