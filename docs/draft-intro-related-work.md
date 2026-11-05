# Draft: Introduction & Related Work

**Working title:** Scenario-Aware Adaptive Large Neighborhood Search for Stochastic Flexible Job Shop Scheduling  
**Venue target:** IJCAI 2027 Main Track (retargetable to a later AAAI)  
**Status:** English academic prose draft for §1–§2. Claims are **honest**: no assertion of superiority over SPM-DAN, CP-stoch, or full-suite significance. Empirical statements that go beyond the current M1 smoke protocol are phrased as *intended* evaluation design, not as completed results.  
**Sources:** `docs/paper-outline-sfjsp-alns.md`, `docs/scenario-aware-alns-design.md`, `results/alns-m1-smoke.md`  
**Date:** 2026-11-05 (Asia/Shanghai)

---

## 中文节注（写作备忘）

- **引言**：问题→缺口（同协议元启发式不足）→方法一句话→贡献三条（框架/搜索–评测分离/相对 PDR 的可复现基线）；**禁止**写“超过 SPM-DAN/CP-stoch”。
- **相关工作**：随机车间 → NCO 车间 → LNS/ALNS → 短 LLM（NS4S 边界）。引用均已核对 DOI/会刊页。
- 冒烟数字（E[Cmax]=640.59 vs MOR 686.97）仅可作 *preliminary* 脚注或附录，勿入主表叙事。

---

## 1. Introduction

Flexible job shop scheduling (FJSP) asks for jointly assigning operations to compatible machines and sequencing them so as to minimize makespan or related criteria. The deterministic problem is already strongly NP-hard; when processing times are uncertain, the planner must reason about a *distribution* of schedules rather than a single nominal Gantt chart. In practice this uncertainty is often handled by sampling processing-time *scenarios* and optimizing a risk-aware scalarization such as expected makespan or Value-at-Risk (VaR). Recent neural combinatorial optimization (NCO) work has shown that deep policies can absorb scenario information and produce strong dispatching decisions under such objectives [Smit et al., 2025]. At the same time, classical priority dispatching rules (PDRs) remain widely used as transparent, training-free baselines, yet they typically leave substantial optimality gaps on stochastic instances.

A complementary line of attack is *search over complete schedules*: destroy–repair metaheuristics such as adaptive large neighborhood search (ALNS) have a long record on routing and scheduling, are easy to instrument, and do not require pretrained weights. However, applying ALNS to *stochastic* FJSP under a protocol that matches modern NCO benchmarks is less developed than the corresponding neural literature. In particular, if the same scenario samples used for search are reused for reporting, the search can overfit the evaluation scenarios and inflate apparent gains. We argue that a carefully engineered, scenario-aware ALNS—with an explicit split between search and evaluation scenario sets—offers a reproducible, interpretable counterpart to scenario-conditioned neural policies on the same instance families and metrics.

This paper develops such a method. We represent a solution by a two-layer encoding consisting of a machine assignment for every operation and a precedence-feasible global operation sequence, decoded by a deterministic earliest-start schedule builder. Stochastic quality is estimated by Monte Carlo evaluation of makespan over sampled processing-time realizations. Destroy and repair operators act on this encoding; several operators are designed to exploit scenario statistics (for example, critical-path structure under sampled times, high cross-scenario variance, or machine-load imbalance). Operator selection follows the classic ALNS roulette scheme with simulated-annealing acceptance. Optionally, a large language model (LLM) may later bias operator choice as an *ablation add-on*; following the spirit of recent LLM-guided neighborhood search for *deterministic* shop scheduling [Zhang et al., 2025], we treat any LLM component as secondary and insist that a strong non-LLM ALNS remain the primary baseline.

**Contributions.** We make three claims at the level of problem framing and method design (empirical claims beyond preliminary smoke tests are deferred to the experimental section once full protocols are complete):

1. **Protocol-aligned metaheuristic for stochastic FJSP.** We formulate scenario-aware ALNS on the same families of stochastic instances and the same E[$C_{\max}$] / VaR reporting conventions popularized by Smit et al. [2025], including held-out evaluation scenario slices drawn from published realization files.
2. **Search–evaluation scenario separation.** We separate the scenario bundle used inside the search loop from the fixed evaluation bundle used for paper tables, reducing the risk that adaptive search overfits the reported metric.
3. **Weight-free, ablatable baseline.** We provide a training-free solver with transparent destroy/repair operators and a clear ablation path (scenario-aware vs.\ random operators; with vs.\ without adaptive weights; optional LLM control), positioned as a competitive *classical* counterpart to NCO rather than as a replacement that already dominates SPM-DAN or CP-based stochastic solvers.

We emphasize what we do *not* claim in the present draft: superiority over SPM-DAN, DAN-stoch, or CP-stoch on full-scale tables; statistical significance on large instance suites; or that LLM guidance improves stochastic schedules. Preliminary smoke runs on three SD3 $10\times5$ instances (evaluation slice of 50 scenarios) show that a minimal SA-ALNS configuration already improves expected makespan relative to MOR under a fixed local protocol; those numbers are diagnostic only and are not presented as submission-ready evidence.

The remainder of the paper is organized as follows. Section 2 reviews stochastic shop scheduling, NCO for job shops, LNS/ALNS, and LLM-assisted combinatorial search. Section 3 formalizes the stochastic FJSP model and evaluation protocol. Section 4 details the encoding, operators, and ALNS loop. Section 5 describes the experimental plan and results. Section 6 concludes.

---

## 2. Related Work

### 2.1 Stochastic and robust shop scheduling

Shop scheduling under uncertainty has been studied for decades through stochastic programming, robust optimization, and simulation–optimization hybrids. When full distributions are unavailable or intractable, *sample average approximation* (SAA) replaces the expectation by an average over a finite scenario set; risk measures such as VaR or conditional VaR are then estimated by order statistics on the same samples. For flexible shops, uncertainty may affect processing times, machine availability, or arrivals. Our setting follows the publicly released stochastic FJSP protocol of Smit et al. [2025]: nominal FJSP instances are paired with banks of processing-time realizations, and solution quality is reported as expected makespan and VaR at a fixed level (e.g., 95\%) on a designated evaluation slice. We adopt this protocol so that metaheuristic and neural methods can be compared under matched data and metrics, rather than under incompatible noise models.

### 2.2 Neural combinatorial optimization for job shops

Learning-based dispatching has advanced rapidly for deterministic JSP and FJSP. Graph neural policies that operate on disjunctive-graph (or heterogeneous graph) states can learn priority rules that generalize across sizes; representative lines include Learning to Dispatch (L2D) for JSP [Zhang et al., 2020] and end-to-end DRL for FJSP with heterogeneous GNNs [Song et al., 2023]. Most of this literature optimizes deterministic makespan. Smit et al. [2025] extend NCO to *stochastic* FJSP via an attention-based Scenario Processing Module (SPM) that embeds sampled scenarios and intervenes on a base policy (SPM-DAN), training under either expected makespan or VaR. Their study also reports classical PDRs and constraint-programming baselines (including stochastic CP variants) on SD1/SD3-style instance sets, and releases code and realization files that we reuse for protocol alignment.

Our work is complementary rather than adversarial in narrative: SPM-DAN learns a *constructive* stochastic policy, whereas we search in the space of *complete* schedules with explicit multi-scenario evaluation. We do not claim, in this draft, to surpass SPM-DAN or CP-stoch; a fair comparison requires matched evaluation sizes (e.g., $K_{\mathrm{eval}}=1000$), wall-clock or decode budgets, and available or re-trained neural weights. Until those experiments are complete, neural and CP entries should be listed as planned baselines or marked unavailable when weights cannot be obtained.

### 2.3 Large neighborhood search and ALNS

Large neighborhood search (LNS) iteratively destroys part of a solution and repairs it with a constructive heuristic or exact subsolver [Shaw, 1998]. Adaptive LNS (ALNS) maintains a portfolio of destroy and repair operators and updates selection weights from recent success [Ropke and Pisinger, 2006]. The framework is attractive for scheduling because operators can encode domain structure (critical paths, bottlenecks, job blocks) while the adaptive layer reallocates effort online. Applications of LNS/ALNS to shop scheduling and related production problems are numerous; our focus is specifically on making destroy/repair *scenario-aware* for stochastic FJSP—e.g., removing operations that are critical under many sampled realizations, or that exhibit high makespan sensitivity across scenarios—and on coupling ALNS with a strict search/eval scenario split. Simulated annealing (or related acceptance rules) is a standard companion to ALNS for escaping local minima; we adopt SA acceptance in our implementation.

### 2.4 LLMs for combinatorial optimization (boundary note)

Large language models have recently been used to propose heuristics, select operators, or guide local search for combinatorial problems. For shop scheduling, NS4S [Zhang et al., 2025] employs LLMs to adjust operation weights that steer neighborhood evaluation on *deterministic* JSP/FJSP (and setup-time variants), with a verification–evolution loop to mitigate hallucination. Our optional LLM module, if activated, would play a narrower role—biasing ALNS operator selection or repair preferences from compact state features—and would be reported only as an ablation on top of a strong non-LLM ALNS. We deliberately do not target the same deterministic FJSP setting as NS4S; the present problem is stochastic, and the primary scientific question is whether explicit scenario search is competitive under the Smit et al. protocol.

---

## References (cited above)

1. **Igor G. Smit, Yaoxin Wu, Pavel Troubil, Yingqian Zhang, and Wim P. M. Nuijten.** Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(25):26678–26687, 2025. DOI: [10.1609/aaai.v39i25.34870](https://doi.org/10.1609/aaai.v39i25.34870). arXiv: [2412.14052](https://arxiv.org/abs/2412.14052).

2. **Junjie Zhang, Canhui Luo, Zhouxing Su, Qingyun Zhang, Zhipeng Lü, Junwen Ding, and Yan Jin.** NS4S: Neighborhood Search for Scheduling Problems Via Large Language Models. In *Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)*, pages 8687–8695, 2025. DOI: [10.24963/ijcai.2025/966](https://doi.org/10.24963/ijcai.2025/966).

3. **Stefan Ropke and David Pisinger.** An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*, 40(4):455–472, 2006. DOI: [10.1287/trsc.1050.0135](https://doi.org/10.1287/trsc.1050.0135).

4. **Paul Shaw.** Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems. In *Principles and Practice of Constraint Programming — CP98*, LNCS 1520, pages 417–431. Springer, 1998. DOI: [10.1007/3-540-49481-2_30](https://doi.org/10.1007/3-540-49481-2_30).

5. **Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, and Chi Xu.** Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 33:1621–1632, 2020.

6. **Wen Song, Xinyang Chen, Qiqiang Li, and Zhiguang Cao.** Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning. *IEEE Transactions on Industrial Informatics*, 19(2):1600–1610, 2023. DOI: [10.1109/TII.2022.3189725](https://doi.org/10.1109/TII.2022.3189725).

---

## Draft hygiene checklist

- [x] No claim of beating SPM-DAN / CP-stoch / DAN-stoch  
- [x] Smoke MOR gap mentioned only as preliminary / diagnostic  
- [x] NS4S cited as deterministic boundary, not as prior stochastic result  
- [x] All numbered refs verified via AAAI/IJCAI/INFORMS/IEEE/Springer pages  
- [ ] After full experiments: rewrite last intro paragraph to cite actual tables; remove “planned baseline” hedges where filled  

