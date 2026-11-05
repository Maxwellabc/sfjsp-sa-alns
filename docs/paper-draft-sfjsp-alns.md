# Scenario-Aware Adaptive Large Neighborhood Search for Stochastic Flexible Job Shop Scheduling

**Draft for ICAPS 2027** (International Conference on Automated Planning and Scheduling)  
**Timing:** Abstract ~Dec 7 2026 / paper ~Dec 14 2026 (UTC−12); empirical tables = **measured P0 20× + full-100 ALNS vs PDRs** (CP-600s still PENDING).  
**Status:** Coherent English manuscript retargeted for a planning & scheduling audience; Method §4 expanded from design doc + `src/sfjsp_alns` (2026-11-05, Asia/Shanghai).  
**Honesty banner:** Tables report **measured** SA-ALNS vs PDRs on (i) **P0 20×** (`results/p0-sd3-10x5.*`, Table 1) with short-budget CP-stoch 60s, and (ii) **full 100×** (`results/full-sd3-10x5.*`, Table 1b). **Realization banks are separate** — the full-100 suite used a newly generated official bank (`NUM_SAMPLES=450`, `seed_datagen=200`); do **not** mix P0 20-inst macros (e.g.\ MOR 654.31) with full-100 macros (e.g.\ MOR 641.89) in one claim. **SPM-DAN remains N/A**. Do **not** claim wins over paper-budget CP (**3600s**) or trained SPM-DAN. **Still in progress:** mid-budget CP-stoch **600s** (`results/p0-cp-stoch-600s*`) — §5.2.2 stays PENDING. Still TODO: K_eval=1000 · paper-budget CP 3600s · multi-seed ≥20-inst destroy ablation.  
**Code / data notes:** `/workspace/fjsp-combo-b/` · `results/p0-sd3-10x5.*` · `results/full-sd3-10x5.*` · `results/p0-cp-stoch.*` · `results/p0-baselines-compare.*`

---

## Abstract

Stochastic flexible job shop scheduling (FJSP) with uncertain processing times is a core planning and scheduling problem: the planner must choose machine assignments and operation sequences that perform well in expectation (or under risk measures such as Value-at-Risk) over sampled processing-time scenarios. Recent neural combinatorial optimization (NCO) methods, notably SPM-DAN [Smit et al., 2025], absorb scenario information into learned dispatching policies but require trained weights. We propose a complementary, **training-free classical** approach for the same stochastic scheduling setting: **scenario-aware adaptive large neighborhood search (SA-ALNS)** over complete schedules. As a first-class methodological contribution, we enforce an explicit **search–evaluation scenario split**—adaptive search may only use a dedicated search scenario bundle, while all reported metrics are computed on a held-out evaluation bundle—so that scenario overfitting does not inflate table results. Solutions use a two-layer encoding (machine assignment and precedence-feasible operation sequence). Destroy operators include random and scenario-aware variants (critical path, high processing-time variance, machine load); repair uses SPT-biased reinsertion with light relocation. On a protocol aligned with Smit et al.—20 SD3 `10×5+mix` instances, three seeds, K_eval=100—SA-ALNS achieves macro-averaged expected makespan **597.16** versus MOR **654.31** (−8.73%) and MWKR **654.18**, winning on 20/20 instances, and reduces macro VaR₉₅ from 762.64 to 683.98. Under the same held-out eval, a short-budget official CP-stoch mean solve (60s, K_cp=10) reaches **618.92** (VaR₉₅ 722.35); SA-ALNS wins on 18/20 instances against that short CP run (−3.51% macro). On the **full 100-instance** SD3 `10×5+mix` suite under the same protocol family but a **separately generated** realization bank (`NUM_SAMPLES=450`, `seed_datagen=200`; not the P0 20-inst bank), SA-ALNS reaches **582.27** versus MOR **641.89** (−9.29%) and MWKR **636.66**, winning on **100/100** instances (VaR₉₅ 666.45 vs MOR 749.98); CP was not run in that pass. **SPM-DAN remains N/A** (no public pretrained FJSP weights). We do **not** claim superiority over Smit et al.’s long-budget CP (e.g., 3600s) or trained neural policies; mid-budget CP-600s (not 3600s) and K_eval=1000 evaluations remain TODO / in progress. SA-ALNS is positioned as a reproducible, weight-free classical counterpart to NCO—not as a neural method—against which neural stochastic FJSP solvers can be compared under matched planning protocols.

---

## 1. Introduction

Flexible job shop scheduling (FJSP) is a canonical automated planning and scheduling task: jointly assign operations to compatible machines and sequence them so as to minimize makespan or related criteria. The deterministic problem is already strongly NP-hard; when processing times are uncertain, the planner must reason about a *distribution* of schedules rather than a single nominal Gantt chart. In practice this uncertainty is often handled by sampling processing-time *scenarios* and optimizing a risk-aware scalarization such as expected makespan or Value-at-Risk (VaR)—a simulation–optimization view familiar to the planning and scheduling community. Recent neural combinatorial optimization (NCO) work has shown that deep policies can absorb scenario information and produce strong dispatching decisions under such objectives [Smit et al., 2025]. At the same time, classical priority dispatching rules (PDRs) remain widely used as transparent, training-free baselines, yet they typically leave substantial optimality gaps on stochastic instances.

A complementary line of attack is *search over complete schedules*: destroy–repair metaheuristics such as adaptive large neighborhood search (ALNS) have a long record on routing and scheduling, are easy to instrument, and do not require pretrained weights [Shaw, 1998; Ropke and Pisinger, 2006]. However, applying ALNS to *stochastic* FJSP under a protocol that matches modern NCO benchmarks is less developed than the corresponding neural literature. In particular, if the same scenario samples used for search are reused for reporting, the search can overfit the evaluation scenarios and inflate apparent gains. We therefore treat an explicit **search–evaluation scenario split** as a first-class methodological contribution for stochastic scheduling: adaptive operator selection and acceptance may only consult a search scenario bundle, while all published metrics are evaluated on a held-out bundle. Under that discipline, a carefully engineered, scenario-aware ALNS offers a reproducible, interpretable *classical* counterpart to scenario-conditioned neural policies on the same instance families and metrics—not a neural method competing for the same training pipeline.

This paper develops such a method. We represent a solution by a two-layer encoding consisting of a machine assignment for every operation and a precedence-feasible global operation sequence, decoded by a deterministic earliest-start schedule builder. Stochastic quality is estimated by Monte Carlo evaluation of makespan over sampled processing-time realizations. Destroy and repair operators act on this encoding; several operators exploit scenario statistics (critical-path structure under sampled times, high cross-scenario variance, or machine-load imbalance). Operator selection follows the classic ALNS roulette scheme with simulated-annealing (SA) acceptance. Optionally, a large language model (LLM) may later bias operator choice as an *ablation add-on*; following the spirit of recent LLM-guided neighborhood search for *deterministic* shop scheduling [Zhang et al., 2025], we treat any LLM component as secondary and insist that a strong non-LLM ALNS remain the primary baseline. **No LLM is used in the experiments of this draft.**

**Contributions.**

1. **Protocol-aligned metaheuristic for stochastic planning/scheduling.** We formulate scenario-aware ALNS for stochastic FJSP on the same SD3-style instance families and E[$C_{\max}$] / VaR reporting conventions popularized by Smit et al. [2025], so that classical search and NCO can be compared under a matched planning protocol (including held-out evaluation scenario slices).
2. **Search–evaluation scenario separation as methodology.** We separate the scenario bundle used inside the search loop from the fixed evaluation bundle used for tables, treating this split as a first-class safeguard against scenario overfitting when reporting stochastic schedule quality.
3. **Weight-free classical baseline (not a neural method) with measured PDR and short-CP gains.** On 20 SD3 `10×5` instances (3 seeds, K_eval=100), SA-ALNS improves macro expected makespan by 8.73% over MOR and wins on all instances under our decoder; vs short-budget CP-stoch (60s) it is 3.51% better macro and wins 18/20. On the full 100-instance suite (same protocol family; **separate** realization generation—do not merge with the 20-inst bank), SA-ALNS reaches 582.27 vs MOR 641.89 (−9.29%) and wins 100/100. A small destroy ablation suggests modest benefit from scenario-aware operators. We position SA-ALNS as a competitive *training-free classical* counterpart to NCO (SPM-DAN)—**not** as a claim that it dominates trained SPM-DAN or paper-budget CP (3600s).

**What we do not claim.** Superiority over SPM-DAN / DAN-stoch (unavailable weights) or over CP-stoch at Smit et al.’s long time limits; that the P0 20-inst and full-100 macros come from the same realization bank (they do not); formal paired significance tests beyond win counts on the 100-instance suite; validity of K_eval=100 as a substitute for K_eval=1000; or that LLM guidance improves stochastic schedules.

The remainder of the paper is organized as follows. Section 2 reviews related work. Section 3 formalizes the problem and protocol. Section 4 details SA-ALNS. Section 5 reports experiments. Section 6 concludes with limitations and TODOs.

---

## 2. Related Work

### 2.1 Stochastic and robust shop scheduling

Shop scheduling under uncertainty has long been studied through stochastic programming, robust optimization, and simulation–optimization hybrids familiar to the planning and scheduling community. When full distributions are unavailable or the expectation is intractable to optimize exactly, *sample average approximation* (SAA) replaces the true expectation by an empirical average over a finite scenario sample and solves the resulting deterministic surrogate [Kleywegt et al., 2002]. Risk measures such as Value-at-Risk (VaR) are likewise estimated from order statistics on the same samples, giving planners a tunable trade-off between average performance and tail makespan. In flexible shops, uncertainty may affect processing times, machine availability, or arrivals; the planner’s decision (machine assignment and sequencing) must then be evaluated by rolling the schedule through many realizations—a simulation–optimization view rather than a single nominal Gantt chart.

Constraint programming (CP) and other OR solvers are a natural counterpart in this setting: a stochastic CP model can optimize a sample-average (or risk) objective over a small scenario set under a wall-clock budget, and Smit et al. [2025] report such CP-stoch baselines alongside neural policies. Our experimental section likewise includes a **short-budget** official CP-stoch mean solve under matched held-out evaluation; longer paper-budget CP (e.g., 3600s) remains future work and is **not** claimed here.

Our empirical setting follows the publicly released stochastic FJSP protocol of Smit et al. [2025]: nominal FJSP instances are paired with banks of processing-time realizations, and solution quality is reported as expected makespan and VaR$_{0.95}$ on a designated evaluation slice. We adopt this protocol so that classical metaheuristics and neural methods can be compared under matched data, metrics, and (when available) decoder stacks, rather than under incompatible noise models.

### 2.2 Neural combinatorial optimization for job shops

Learning-based dispatching has advanced rapidly for deterministic JSP and FJSP. Graph neural policies on disjunctive or heterogeneous graph states can learn priority rules that generalize across sizes; representative lines include Learning to Dispatch (L2D) for JSP [Zhang et al., 2020] and end-to-end deep reinforcement learning for FJSP [Song et al., 2023]. Most of this literature optimizes deterministic makespan. Smit et al. [2025] extend NCO to *stochastic* FJSP via an attention-based Scenario Processing Module (SPM) that embeds sampled scenarios and intervenes on a base policy (SPM-DAN), training under expected makespan or VaR. Their study also reports classical PDRs and constraint-programming baselines (including stochastic CP variants) on SD1/SD3-style instance sets, and releases code and realization recipes that we reuse for protocol alignment. **SPM-DAN is the protocol reference** for our instance families and metrics; it is **not** a runnable baseline in the present draft.

Our work is complementary rather than adversarial: SPM-DAN learns a *constructive* stochastic policy that builds a schedule decision-by-decision, whereas we search in the space of *complete* schedules with explicit multi-scenario evaluation after each destroy–repair move. We report a short-budget CP-stoch mean baseline under matched held-out eval; **SPM-DAN stays N/A** experimentally without public pretrained FJSP weights. Matched long-budget CP and trained SPM-DAN comparisons remain future work and are explicitly excluded from our claims.

### 2.3 Large neighborhood search and ALNS

Large neighborhood search (LNS) iteratively destroys part of a solution and repairs it with a constructive heuristic or exact subsolver [Shaw, 1998]. Adaptive LNS (ALNS) maintains a portfolio of destroy and repair operators and updates selection weights from recent success [Ropke and Pisinger, 2006]. Surveys of LNS/ALNS emphasize that the framework has shown strong results not only on vehicle routing but also on a range of *scheduling* and production problems, precisely because operators can encode domain structure (critical paths, bottlenecks, job blocks) while the adaptive layer reallocates effort online [Pisinger and Ropke, 2010]. Simulated annealing (or related acceptance rules) is a standard companion for escaping local minima; we adopt SA acceptance in our implementation.

For *stochastic* FJSP under a modern NCO-aligned protocol, two design choices matter beyond the classical ALNS skeleton. First, destroy operators should be *scenario-aware*: rather than removing operations only by random or nominal criteria, they can target operations that are critical under sampled realizations, that exhibit high cross-scenario processing-time variance, or that sit on overloaded machines under mean scenario load. Second, adaptive search must not be allowed to fit the same scenarios that appear in paper tables. We therefore couple ALNS with an explicit **search–evaluation scenario split**: operator scoring and acceptance consult only a search bundle, while all reported E[$C_{\max}$] / VaR numbers are computed on a held-out evaluation bundle. That split is our methodological angle for ICAPS readers—treating scenario hold-out as first-class methodology for stochastic schedule search, not as an afterthought of evaluation hygiene.

### 2.4 LLMs for combinatorial optimization (boundary)

For shop scheduling, NS4S [Zhang et al., 2025] employs LLMs to adjust operation weights that steer neighborhood evaluation on *deterministic* JSP/FJSP. An optional LLM module in our pipeline would only bias ALNS operator selection and would be reported as an ablation on top of a strong non-LLM baseline. We do not target the deterministic setting of NS4S; the present problem is stochastic, and **no LLM is used in the experiments of this draft**.

### 2.5 Classical vs.\ NCO positioning (for ICAPS readers)

Taken together, Sections 2.1–2.3 locate SA-ALNS as a **weight-free, reproducible classical** counterpart to scenario-conditioned NCO on the same stochastic FJSP protocol: complete-schedule search with scenario-aware neighborhoods and a strict search–eval split, rather than a learned constructive policy. SPM-DAN remains the protocol reference and stays **N/A** experimentally without public FJSP weights; short-budget CP-stoch is reported as an honest OR baseline, not as a stand-in for paper-budget (e.g., 3600s) CP. We do **not** claim to beat trained SPM-DAN or long-budget CP. The intended contribution for a planning and scheduling audience is a transparent, ablatable metaheuristic baseline against which neural stochastic FJSP solvers can be compared under matched instances, metrics, and scenario discipline.

---

## 3. Problem Definition and Evaluation Protocol

### 3.1 Stochastic FJSP

An FJSP instance comprises $J$ jobs and $M$ machines. Job $j$ has a sequence of operations; operation $i$ may be processed on a subset of machines with nominal processing time $p_{im}>0$ when machine $m$ is compatible (else $p_{im}=0$). Under uncertainty, each scenario $s$ replaces the processing-time matrix by a realization $p^{s}$. A solution $x$ specifies (i) a machine assignment for every operation and (ii) a global operation sequence respecting intra-job precedence. Given $(x,p^{s})$, a deterministic earliest-start decoder yields makespan $C_{\max}(x;s)$.

### 3.2 Objectives

For a scenario set $S=\{s_1,\ldots,s_K\}$,

$$
\hat{\mathbb{E}}[C_{\max}](x)=\frac{1}{K}\sum_{k=1}^{K} C_{\max}(x;s_k),
\qquad
\widehat{\mathrm{VaR}}_{\alpha}(x)=\mathrm{Percentile}_{100\alpha}\{C_{\max}(x;s_k)\}_{k=1}^{K},
$$

with default $\alpha=0.95$. Search may optimize either objective; we report both on the evaluation set. In P0 experiments the search objective is expected makespan.

### 3.3 Search vs.\ evaluation scenarios

| Role | Slice used in P0 | Purpose |
|------|------------------|---------|
| Search | `[0:50]` ($K_{\mathrm{search}}=50$) | ALNS acceptance and operator scores |
| Evaluation | `[250:350]` ($K_{\mathrm{eval}}=100$) | Paper tables only |

**Rule:** the evaluation slice is never used for tuning or weight updates. **TODO:** align $K_{\mathrm{eval}}$ with Smit et al.\ (often 1000) once larger realization banks are generated.

---

## 4. Method: Scenario-Aware ALNS

We search over *complete* schedules rather than learning a constructive dispatching policy. The solver, denoted **SA-ALNS**, couples a two-layer FJSP encoding with adaptive large neighborhood search [Ropke and Pisinger, 2006], simulated-annealing acceptance, and destroy operators that read statistics from a dedicated *search* scenario bundle. Evaluation metrics are computed exclusively on a disjoint held-out bundle (Section 3.3). No neural weights and no LLM calls are required for the configuration used in our experiments.

### 4.1 Solution encoding and deterministic decoder

Let $N$ be the number of operations and $M$ the number of machines. A solution is
\[
x=(\mu,\pi),\qquad
\mu\in\{0,\ldots,M-1\}^{N},\quad
\pi\in\mathrm{Perm}(\{0,\ldots,N-1\}),
\]
where $\mu_i$ is the machine assigned to operation $i$ (restricted to the compatible set $\mathcal{M}_i=\{m:p_{im}>0\}$) and $\pi$ is a global operation order that respects intra-job precedence: if operations $i$ and $i'$ belong to the same job and $i$ precedes $i'$ in the technological route, then $i$ appears before $i'$ in $\pi$.

Given a processing-time matrix $p$ (nominal or scenario-specific), the decoder builds an earliest-start schedule by scanning $\pi$ in order. For each operation $i=\pi_t$ on machine $m=\mu_i$,
\[
\mathrm{start}_i=\max\bigl(R^{\mathrm{job}}_{j(i)},\,R^{\mathrm{mch}}_{m}\bigr),\qquad
\mathrm{end}_i=\mathrm{start}_i+p_{im},
\]
then updates job and machine ready times. The makespan is $C_{\max}(x;p)=\max_i\mathrm{end}_i$. The map $(x,p)\mapsto C_{\max}$ is deterministic, which is essential for reproducible multi-scenario evaluation and for fair comparison across methods that share the same decoder.

If $p_{im}\le 0$ for the assigned machine under a scenario (incompatibility), we treat the decode as infeasible by returning a large sentinel cost; under the official realization generators used here, compatible machines remain compatible across scenarios.

### 4.2 Scenario evaluation and search objective

For a scenario bundle $S=\{p^{(1)},\ldots,p^{(K)}\}$ we estimate
\[
\hat{\mathbb{E}}[C_{\max}](x;S)=\frac1K\sum_{k=1}^{K}C_{\max}(x;p^{(k)}),
\qquad
\widehat{\mathrm{VaR}}_{\alpha}(x;S)=\mathrm{Percentile}_{100\alpha}\bigl\{C_{\max}(x;p^{(k)})\bigr\}.
\]
During search we optimize a scalar objective $\mathrm{obj}\in\{\hat{\mathbb{E}},\widehat{\mathrm{VaR}}_{\alpha}\}$ on the **search** bundle $S_{\mathrm{search}}$ (P0 uses expectation). After search terminates, we report both $\hat{\mathbb{E}}$ and $\widehat{\mathrm{VaR}}_{0.95}$ on the **evaluation** bundle $S_{\mathrm{eval}}$, which never enters operator scoring or acceptance.

To control decode cost inside neighborhoods, expensive local probes may use a random subsample of $K_{\mathrm{fast}}$ search scenarios (common random numbers within an iteration). Full-bundle evaluation is used for ALNS acceptance on $S_{\mathrm{search}}$ and for refining the incumbent.

### 4.3 Initialization from priority dispatching rules

We construct four feasible seeds by rolling out classical PDRs—FIFO, MOR (most operations remaining), SPT, and MWKR (most work remaining)—on the *nominal* processing-time matrix $p^{\mathrm{nom}}$, using a discrete-event style policy that always schedules a ready job's next operation onto a compatible machine with earliest feasible start (ties broken by a fixed seed). Each seed is scored with $\mathrm{obj}(\,\cdot\,;S_{\mathrm{search}})$; ALNS starts from the best. This best-of-PDR warm start is cheap, weight-free, and matches the PDR baselines reported in Section 5 under the same decoder.

Optional warm-starts from deterministic high-quality schedules (e.g., NS4S [Zhang et al., 2025] or DAN [Song et al., 2023]) can be mapped into $(\mu,\pi)$ but are **not** used in the present experiments, so that measured gains are attributable to stochastic search rather than a stronger deterministic oracle.

### 4.4 Destroy operators

At destruction intensity $q\sim U(q_{\min},q_{\max})$, we select a set $\Omega$ of $k=\mathrm{round}(qN)$ operations (capped at $\lfloor N/2\rfloor$), clear their machine assignments ($\mu_i\leftarrow\text{unassigned}$ for $i\in\Omega$), and **retain** the full sequence $\pi$. Keeping $\pi$ avoids breaking precedence during destroy and makes repair primarily a machine-reassignment / light-relocate problem—important for decoder stability in early prototypes and for the P0 configuration.

**D1 — random ops.** Sample $\Omega$ uniformly. This non-informed operator provides diversification and anchors the ablation baseline.

**D3 — critical path (scenario-aware).** Draw one scenario $p\sim S_{\mathrm{search}}$, decode $(x,p)$ to obtain completion times, and preferentially include operations whose completion is close to $C_{\max}(x;p)$, filling the remainder of $\Omega$ by decreasing completion time. The intent is to free operations that currently sit on (or near) a makespan-critical chain under a concrete realization, rather than under nominal times alone.

**D4 — high-variance ops (scenario-aware).** For each operation $i$ with assigned machine $m=\mu_i$, compute the coefficient of variation of $\{p^{(k)}_{im}\}_{k}$ over $S_{\mathrm{search}}$. Select the $k$ highest-CV operations. These operations are those whose assigned processing times fluctuate most across scenarios and are therefore natural targets when optimizing risk-aware or expectation objectives under heterogeneous noise.

**D5 — machine load (scenario-aware).** Estimate mean load of machine $m$ as $\sum_{i:\mu_i=m}\bar p_{im}$ with $\bar p$ the mean processing-time matrix over $S_{\mathrm{search}}$. Rank operations by the load of their current machine and clear the top $k$. This targets congested resources that dominate makespan under average scenario load.

Operators D2 (sequence-window shuffle) and D6 (job-block removal) are reserved in the codebase but disabled in P0. The adaptive layer (Section 4.6) learns relative frequencies among the enabled destroy names online.

### 4.5 Repair: SPT-biased reinsertion

**R4 — `spt_biased`.** Let $\bar p$ be the mean processing-time matrix over $S_{\mathrm{search}}$. For each cleared operation $i\in\Omega$ (random order):

1. **Machine fill.** Tentatively assign $i$ to $\arg\min_{m\in\mathcal{M}_i}\bar p_{im}$ (SPT on mean scenario times).
2. **Local improve.** Consider the top few machines by $\bar p_{im}$ and a small set of precedence-feasible insertion positions for $i$ in $\pi$ (bounded by the positions of the job predecessor and successor). Score candidates by mean $C_{\max}$ on a $K_{\mathrm{fast}}$ subsample of $S_{\mathrm{search}}$, and keep the best.

Richer repairs (greedy mean/VaR insertion, regret-$k$, beam search) are natural extensions; P0 uses only R4 for speed and simplicity. After repair, a short **machine hill-climb** randomly reassigns individual operations to alternate compatible machines when the $K_{\mathrm{fast}}$ score improves, providing a cheap intensification step independent of destroy choice.

### 4.6 Adaptive search with simulated annealing

Algorithmically, SA-ALNS maintains weights $w_d$ and $w_r$ over destroy and repair names (initially uniform). At iteration $t$:

1. Sample destroy $d$ and repair $r$ by roulette wheel on current weights; sample intensity $q$.
2. Form a neighbor $y$ by $d$-destroy then $r$-repair, followed by the machine hill-climb.
3. Let $\Delta=\mathrm{obj}(y;S_{\mathrm{search}})-\mathrm{obj}(x;S_{\mathrm{search}})$. Accept $y$ if $\Delta<0$, or with probability $\exp(-\Delta/T)$ otherwise (Simulated Annealing). Temperature is initialized as $T_0=\tau\cdot\mathrm{obj}(x_0)$ and cooled multiplicatively, $T\leftarrow \rho T$ each iteration.
4. If accepted, update operator scores. Whenever the accepted solution improves the incumbent on a refine view of the search scenarios, record a new best $x^\star$ and award a larger score bonus to $(d,r)$.
5. Every $L$ iterations (a *segment*), update weights by the standard ALNS reaction rule
   \[
   w \leftarrow (1-\lambda)\,w+\lambda\cdot\frac{\mathrm{score}}{\#\mathrm{uses}},
   \]
   reset segment scores, and refresh the $K_{\mathrm{fast}}$ index set.

Search stops at a maximum iteration count or a wall-clock limit. **Only then** do we evaluate $x^\star$ on $S_{\mathrm{eval}}$ for reporting. This separation is the methodological counterpart to holding out a test set in supervised learning: adaptive weights must not be fitted to the numbers that appear in paper tables.

**P0 hyperparameters (SD3 $10\times5$).** Max iterations $300$ or $40$s; $q\in[0.08,0.22]$; $K_{\mathrm{fast}}=10$; cooling $\rho=0.995$; reaction $\lambda=0.25$; segment length $40$; hill-climb tries $20$ (half that after each neighbor). Destroy portfolio $\{D1,D3,D4,D5\}$; repair $\{R4\}$.

### 4.7 Complexity remarks

Each full-bundle evaluation costs $O(K\cdot N)$ earliest-start updates (linear in operations per scenario). With $K_{\mathrm{search}}=50$ and a few hundred iterations, decode count—not asymptotic big-O—is the practical budget. Fast scoring with $K_{\mathrm{fast}}\ll K_{\mathrm{search}}$ inside repair and hill-climb keeps intensification affordable. Fair future comparisons to CP-stoch or SPM-DAN should state wall-clock, thread count, and decode (or inference) budgets explicitly (**TODO** in Section 5).

### 4.8 Optional LLM-guided operator selection (out of scope for P0 tables)

As a future ablation, an LLM may, each segment, map compact numeric features (machine loads, critical-path length, scenario CV summary, recent operator scores) to a preferred $(d,r)$ pair or a $q$ band, with deterministic roulette fallback on failure [cf.\ Zhang et al., 2025]. We deliberately **do not** ask an LLM to emit complete schedules. Mode C (LLM disabled) is the strong baseline reported in Section 5; any LLM mode must be shown as an incremental ablation, never as the sole experimental condition.

---

## 5. Experiments

All **filled** numbers below are from measured artifacts dated 2026-11-05 (Asia/Shanghai):
- **P0 20× / CP-60s:** `results/p0-sd3-10x5-raw.csv`, `results/p0-sd3-10x5-summary.json`, `results/p0-ablation-destroy.csv`, `results/p0-cp-stoch-summary.json`, `results/p0-baselines-compare.md` → Tables 1–2.
- **Full 100× ALNS vs PDRs:** `results/full-sd3-10x5-raw.csv`, `results/full-sd3-10x5-summary.json`, `results/full-sd3-10x5.md` → Table 1b.

**Bank separation (critical):** the full-100 realization bank was newly generated under the official path (`NUM_SAMPLES=450`, `seed_datagen=200`) and is **not identical** to the P0 20-instance / earlier `smoke_SD3` bank. Keep Table 1 and Table 1b as **separate** tables; never merge macros across banks in one claim (e.g., do not equate P0 MOR 654.31 with full-100 MOR 641.89).
**Table 1 / Table 2** stay the measured 20× / CP-60s baseline. **CP-600s** (§5.2.2) remains PENDING — do **not** invent macros from partial logs.

### 5.1 Setup

**Instances (measured).** (i) First **20** of SD3 `10x5+mix` (Smit et al.\ release) — Tables 1–2 / CP-60s. (ii) Full size class **100** (`10x5+mix_001`…`_100`) — Table 1b (ALNS vs PDRs only; CP not in that pass).

**Realizations.** Official random lognormal CoV recipe (`seed_datagen=200`), 450 samples per file under `stochastic_realizations/random/SD3/10x5+mix/`. Protocol-compatible with Smit et al. **Important:** the bank used for the **full-100** pass was **newly generated** under that path and is **not identical** to the P0 20-instance bank (nor to the earlier three-instance `smoke_SD3` bank). Do **not** mix P0, full-100, and smoke macros in one claim or one table.

**Methods executed (filled tables).** FIFO, MOR, SPT, MWKR (in-package decoder); SA-ALNS with destroy `{D1,D3,D4,D5}` and repair R4; **CP-stoch short (60s)** — official OR-Tools stochastic mean model, K_cp=10, time_limit=60s, schedule re-decoded on `[250:350]`.

**Methods completed (full-100).** Full-100 SA-ALNS vs PDRs (`scripts/alns_full_sd3_10x5.py` → `results/full-sd3-10x5.*`; wall ≈4111s; max_iter=300). **Methods in progress / reserved (not filled).** **CP-stoch mid-budget (600s)** on the same 20-instance P0 slice (`results/p0-cp-stoch-600s*`). Label 600s as a **longer short-budget / mid-budget** reference — **not** Smit et al.’s paper 3600s budget.

**Methods N/A.** **SPM-DAN / DAN-stoch** — no pretrained FJSP checkpoints in the public repository; short CPU retrain would be unfair and was not run.

**Seeds.** $\{50,51,52\}$ for PDRs/ALNS; CP is a single deterministic run per instance. Macro-average over 20 instances (Tables 1–2) or 100 instances (Table 1b). ALNS P0 suite $\approx 632$s; full-100 ALNS suite $\approx 4111$s; CP-60s suite $\approx 1215$s.

### 5.2 Main results

**Table 1.** Macro averages on 20× SD3 `10x5+mix`, $K_{\mathrm{eval}}=100$, 3 seeds. Gap relative to MOR.

| Method | E[$C_{\max}$] | VaR$_{95}$ | vs MOR (%) |
|--------|-------------:|-----------:|-----------:|
| FIFO | 661.11 | 770.34 | +1.04 |
| MOR | 654.31 | 762.64 | 0.00 |
| SPT | 823.84 | 943.42 | +25.91 |
| MWKR | 654.18 | 763.53 | −0.02 |
| **SA-ALNS** | **597.16** | **683.98** | **−8.73** |
| CP-stoch (60s, K=10) | 618.92 | 722.35 | −5.41 |
| SPM-DAN | N/A | N/A | — |

SA-ALNS improves expected makespan by 57.15 absolute points versus MOR (−8.73%) and reduces macro VaR$_{95}$ from 762.64 to 683.98. MWKR is essentially tied with MOR among PDRs; SPT is much worse. Short-budget CP-stoch reaches 618.92 (−5.41% vs MOR) but remains 3.51% above SA-ALNS; **not** comparable to Smit et al.’s 3600s CP. **Table 1 stays the measured 20× / CP-60s baseline** on the P0 bank; see **Table 1b** for the full-100 suite on a **separate** realization bank. CP-600s remains PENDING (§5.2.2).

**Win counts (20× only).** Seed-mean E[$C_{\max}$]: SA-ALNS beats MOR on **20/20** instances; (instance, seed) grain **60/60**. Versus CP-stoch-60s: **18/20** (losses on `004`, `012`).

**Table 2.** Per-instance seed-mean E[$C_{\max}$].

| Instance | MOR | SA-ALNS | Δ |
|----------|----:|--------:|--:|
| 10x5+mix_001 | 741.66 | 654.59 | −87.07 |
| 10x5+mix_002 | 679.74 | 641.65 | −38.08 |
| 10x5+mix_003 | 665.37 | 616.39 | −48.97 |
| 10x5+mix_004 | 635.13 | 571.57 | −63.56 |
| 10x5+mix_005 | 583.81 | 551.30 | −32.51 |
| 10x5+mix_006 | 612.00 | 562.26 | −49.74 |
| 10x5+mix_007 | 558.59 | 517.34 | −41.24 |
| 10x5+mix_008 | 843.46 | 710.47 | −132.99 |
| 10x5+mix_009 | 717.59 | 661.08 | −56.51 |
| 10x5+mix_010 | 649.39 | 592.84 | −56.55 |
| 10x5+mix_011 | 575.36 | 524.18 | −51.19 |
| 10x5+mix_012 | 635.18 | 609.64 | −25.55 |
| 10x5+mix_013 | 676.73 | 638.25 | −38.49 |
| 10x5+mix_014 | 697.12 | 636.77 | −60.36 |
| 10x5+mix_015 | 643.11 | 562.24 | −80.87 |
| 10x5+mix_016 | 650.81 | 596.56 | −54.26 |
| 10x5+mix_017 | 639.57 | 583.44 | −56.14 |
| 10x5+mix_018 | 641.25 | 579.38 | −61.87 |
| 10x5+mix_019 | 636.23 | 583.36 | −52.87 |
| 10x5+mix_020 | 604.09 | 549.96 | −54.12 |

### 5.2.1 Full 100× SD3 `10×5` (ALNS vs PDRs) — measured

**Status:** **Done** — finished artifacts `results/full-sd3-10x5-raw.csv`, `results/full-sd3-10x5-summary.json`, `results/full-sd3-10x5.md` (wall ≈4111s; max_iter=300; CP **not** run in this pass).

> **Bank separation (read before citing).** Realization bank newly generated under the official path (`NUM_SAMPLES=450`, `seed_datagen=200`). It is **not identical** to the P0 20-instance bank used in Table 1 / Table 2, nor to `smoke_SD3`. Keep **Table 1 (20-inst, P0 bank)** and **Table 1b (100-inst, full-100 bank)** as **separate** tables. Do **not** mix banks in one claim (e.g., do not present MOR 654.31 from Table 1 and MOR 641.89 from Table 1b as the same experiment).

**Table 1b.** Macro averages on **100×** SD3 `10x5+mix`, protocol family matched to P0 ($K_{\mathrm{eval}}=100$, eval `[250:350]`, search `[0:50]`, seeds $\{50,51,52\}$, in-package decoder). Gap relative to MOR. Source: `results/full-sd3-10x5.*`.

| Method | E[$C_{\max}$] | VaR$_{95}$ | vs MOR (%) |
|--------|-------------:|-----------:|-----------:|
| FIFO | 646.99 | 754.60 | +0.80 |
| MOR | 641.89 | 749.98 | 0.00 |
| SPT | 794.03 | 911.95 | +23.70 |
| MWKR | 636.66 | 741.63 | −0.81 |
| **SA-ALNS** | **582.27** | **666.45** | **−9.29** |
| SPM-DAN | N/A | N/A | — |
| CP-stoch | N/A (not in this pass) | — | — |

**Win counts (100×, seed-mean E[$C_{\max}$]).** SA-ALNS beats MOR on **100/100** instances; also beats MWKR on **100/100**. Absolute gap vs MOR: 59.62 points (−9.29%). Macro VaR$_{95}$ falls from 749.98 (MOR) to 666.45. Among PDRs, MWKR is slightly ahead of MOR (−0.81%); SPT remains much worse. These macros must **not** be averaged or pooled with Table 1’s 20-inst numbers.

### 5.2.2 Reserved — CP-stoch mid-budget 600s (PENDING; not 3600s)

**Status:** **PENDING** — live run in progress (`results/p0-cp-stoch-600s-run.log`, out-prefix `p0-cp-stoch-600s`). Target: same 20-instance P0 slice, official OR-Tools stochastic mean model, K_cp=10, **time_limit=600s**, re-decode on `[250:350]`.

**Labeling:** call this a **longer short-budget / mid-budget** CP reference. It is **not** Smit et al.’s paper-budget CP (**3600s**). Do **not** claim 3600s wins from a 600s run.

**Table 1c (reserved).** Macro E[$C_{\max}$] / VaR$_{95}$ for CP-stoch-600s vs Table 1 SA-ALNS / CP-60s (20×).

| Method | Budget | E[$C_{\max}$] | VaR$_{95}$ | vs SA-ALNS | vs CP-60s |
|--------|--------|-------------:|-----------:|-----------:|----------:|
| SA-ALNS (Table 1) | ~10s/inst (P0) | **597.16** | **683.98** | 0.00 | — |
| CP-stoch (Table 1) | 60s | 618.92 | 722.35 | +3.64% | 0.00 |
| CP-stoch mid | **600s** | — PENDING — | — | — | — |
| CP-stoch paper | **3600s** | — TODO — | — | — | — |
| SPM-DAN | — | N/A | N/A | — | — |

**Fold-in rule:** fill the 600s row only from finished `results/p0-cp-stoch-600s*` summary; keep the 3600s row as TODO.

### 5.3 Ablation (probe-only — limited claims)

**Honesty.** Table 3 is a **probe**, not a definitive operator study: **5 instances × seed 50 only**, same short ALNS budget as P0. It can motivate keeping scenario-aware destroys in the portfolio; it **cannot** support strong claims that D3/D4/D5 are necessary or uniformly helpful. **TODO before strong operator claims:** multi-seed ablation on **≥20 instances** (ideally the full P0 20× with seeds $\{50,51,52\}$, and later a slice of the 100× suite), with per-operator contribution or leave-one-out designs as needed.

**Table 3.** Five instances, seed 50, same budget: D1-only vs full destroy set (probe).

| Variant | Destroy set | Mean E[$C_{\max}$] |
|---------|-------------|-------------------:|
| D1 only | `random_ops` | 610.48 |
| Full (P0) | D1+D3+D4+D5 | 604.19 |

Average improvement ≈ 6.29 points (−1.03% vs D1-only). Gains are uneven across the five instances. Until the multi-seed ≥20-inst ablation is done, prose should say “suggests a modest benefit” / “probe evidence,” not “scenario-aware operators are validated.”

### 5.4 Discussion

Under a matched in-package decoder, scenario-aware SA-ALNS consistently improves over strong PDRs on the P0 20-instance slice and also beats a short-budget official CP-stoch mean solve (60s, K_cp=10) on 18/20 instances. On the full 100-instance suite (same protocol family; **separate** realization bank), the PDR gap holds at −9.29% vs MOR with 100/100 wins—confirming the ~9% effect under a broader instance set without merging banks with Table 1. Absolute values are internally consistent within each bank but must not be pooled across banks, and may differ from official torch-env PDR numbers on other realization banks. Short ALNS budgets (~10s / max_iter=300) already suffice for clear PDR gaps; time–quality curves, the CP-600s fold-in (§5.2.2), and paper-matched CP (3600s) remain future / in-progress work. SPM-DAN cannot yet be scored without released weights. Destroy-operator evidence remains probe-level (§5.3).

### 5.5 Honesty / reporting rules

These rules govern every empirical claim in this draft (and any Canvas sync of the same tables):

1. **Search–evaluation scenario split.** Adaptive search and acceptance may use only $S_{\mathrm{search}}$ (`[0:50]` in P0). All reported E[$C_{\max}$] / VaR$_{95}$ use $S_{\mathrm{eval}}$ (`[250:350]`). Never retune on the eval slice.
2. **$K_{\mathrm{eval}}=100$ vs TODO 1000.** Filled tables use $K_{\mathrm{eval}}=100$. Matching Smit et al.’s larger eval ($K_{\mathrm{eval}}=1000$) is **TODO** once larger realization banks exist; do not treat 100 as a substitute for 1000 in camera-ready claims.
3. **SPM-DAN = N/A.** No public pretrained FJSP weights → no SPM-DAN column numbers, no “beats SPM-DAN” claims, no unfair short retrain.
4. **Distinguish CP budgets.** **CP-60s** = measured short-budget (Table 1). **CP-600s** = mid-budget / longer short-budget, reserved (§5.2.2), **not** paper 3600s. **CP-3600s** = Smit-style paper budget, still **TODO**. Never upgrade a 60s or 600s result into a 3600s claim.
5. **No cherry-picking from unfinished runs.** Full-100 ALNS is **filled** from finished `results/full-sd3-10x5.*` (§5.2.1). CP-600s may still be running; partial log lines are **not** macros — fill §5.2.2 only from finished summary artifacts. Table 1 / Table 2 stay the measured 20× / CP-60s baseline on the P0 bank.
6. **Separate realization banks.** Table 1 (P0 20×) and Table 1b (full 100×) use **different** generated realization banks under the same official recipe. Never merge, average, or equate macros across those tables in one claim.
7. **Ablation scope.** Table 3 is probe-only (5× seed 50). Strong operator claims require the multi-seed ≥20-inst ablation TODO.

---

## 6. Conclusion and Limitations

We presented scenario-aware SA-ALNS for stochastic FJSP under a Smit-aligned scenario protocol, with search–evaluation separation as a methodological safeguard and transparent destroy/repair operators suited to a planning and scheduling audience. Measured P0 results show clear gains over FIFO/MOR/SPT/MWKR on 20 SD3 `10×5` instances (E[$C_{\max}$] 597.16 vs MOR 654.31) and a 3.51% macro edge over short-budget CP-stoch (618.92). On the full 100-instance suite (separate realization bank), SA-ALNS reaches 582.27 vs MOR 641.89 (−9.29%) with 100/100 wins. The approach is training-free and ablatable—a weight-free classical counterpart to NCO (SPM-DAN), not a neural method—providing a reproducible reference point for stochastic scheduling research.

**Limitations and TODO checklist (submission blockers in bold):**

| Item | Status |
|------|--------|
| 20× SD3 10×5, 3 seeds, vs PDRs | Done (Table 1–2) |
| D3–D5 + small ablation | Done (**probe only**, Table 3; 5 inst × seed 50) |
| **Full 100× SD3 10×5** (SA-ALNS vs PDRs) | **Done** (Table 1b; separate realization bank from P0; CP not in that pass) |
| **K_eval=1000** | **TODO** |
| **SPM-DAN / DAN-stoch** | **N/A blocked** (no public FJSP weights) |
| CP-stoch short (60s, K=10, mean) | Done (Table 1) |
| **CP-stoch mid-budget (600s)** | **IN PROGRESS** — reserved §5.2.2; label mid-budget, **not** 3600s |
| **CP-stoch paper budget (3600s)** | **TODO** (do not claim from 60s/600s) |
| Larger SD3 sizes (20×5, 15×10, …) | TODO |
| Multi-seed ≥20-inst destroy ablation | **TODO** (required before strong operator claims) |
| LLM operator selection ablation | TODO (optional) |

Until remaining bold items are filled, the manuscript should be framed for **ICAPS 2027** as a strong PDR-beating metaheuristic study in stochastic planning/scheduling (now with measured 20× and 100× PDR tables on **separate** banks), with a matched short-CP reference—not as a complete challenge to Smit et al.’s neural policies or long-budget CP (3600s). Abstract (~Dec 7 2026) and paper (~Dec 14 2026, UTC−12) deadlines govern the remaining empirical TODOs (CP-600s mid-budget fold-in, K_eval=1000, paper-budget CP 3600s, multi-seed ≥20-inst ablation); **SPM-DAN remains N/A** without public FJSP weights. Do not invent macros from partial CP-600s logs.

### Reproducibility

```bash
cd /workspace/fjsp-combo-b && source .venv/bin/activate
SFJSP_REAL_LIMIT=20 PYTHONPATH=baselines/NCO-for-Stochastic-FJSP/FJSP-DRL \
  python scripts/generate_sd3_10x5_realizations.py
PYTHONPATH=src python scripts/alns_p0_sd3_10x5.py
PYTHONPATH=src python scripts/run_cp_stoch_p0.py
```

---

## References

1. Igor G. Smit, Yaoxin Wu, Pavel Troubil, Yingqian Zhang, and Wim P. M. Nuijten. Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(25):26678–26687, 2025. DOI: [10.1609/aaai.v39i25.34870](https://doi.org/10.1609/aaai.v39i25.34870). arXiv: [2412.14052](https://arxiv.org/abs/2412.14052).

2. Junjie Zhang, Canhui Luo, Zhouxing Su, Qingyun Zhang, Zhipeng Lü, Junwen Ding, and Yan Jin. NS4S: Neighborhood Search for Scheduling Problems Via Large Language Models. In *Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)*, pages 8687–8695, 2025. DOI: [10.24963/ijcai.2025/966](https://doi.org/10.24963/ijcai.2025/966).

3. Stefan Ropke and David Pisinger. An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows. *Transportation Science*, 40(4):455–472, 2006. DOI: [10.1287/trsc.1050.0135](https://doi.org/10.1287/trsc.1050.0135).

4. Paul Shaw. Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems. In *Principles and Practice of Constraint Programming — CP98*, LNCS 1520, pages 417–431. Springer, 1998. DOI: [10.1007/3-540-49481-2_30](https://doi.org/10.1007/3-540-49481-2_30).

5. Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, and Chi Xu. Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 33:1621–1632, 2020.

6. Wen Song, Xinyang Chen, Qiqiang Li, and Zhiguang Cao. Flexible Job-Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning. *IEEE Transactions on Industrial Informatics*, 19(2):1600–1610, 2023. DOI: [10.1109/TII.2022.3189725](https://doi.org/10.1109/TII.2022.3189725).

7. Anton J. Kleywegt, Alexander Shapiro, and Tito Homem-de-Mello. The Sample Average Approximation Method for Stochastic Discrete Optimization. *SIAM Journal on Optimization*, 12(2):479–502, 2002. DOI: [10.1137/S1052623499363220](https://doi.org/10.1137/S1052623499363220).

8. David Pisinger and Stefan Ropke. Large Neighborhood Search. In Michel Gendreau and Jean-Yves Potvin (eds.), *Handbook of Metaheuristics*, 2nd edition, pages 399–419. Springer, 2010. DOI: [10.1007/978-1-4419-1665-5_13](https://doi.org/10.1007/978-1-4419-1665-5_13).

---

*Assembled draft for ICAPS 2027. Empirical claims restricted to measured P0 + full-100 CSV/JSON (banks kept separate). Marked TODOs must be resolved before camera-ready submission.*
