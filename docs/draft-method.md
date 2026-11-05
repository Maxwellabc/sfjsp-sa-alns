# Draft: Method — Scenario-Aware ALNS for Stochastic FJSP

**Working title:** Scenario-Aware Adaptive Large Neighborhood Search for Stochastic Flexible Job Shop Scheduling  
**Venue target:** IJCAI 2027 (retargetable to AAAI)  
**Status:** Polished English Method section aligned with `docs/scenario-aware-alns-design.md` and `src/sfjsp_alns/` (as of 2026-11-05, Asia/Shanghai).  
**Merged into:** `docs/paper-draft-sfjsp-alns.md` §4  

---

## 中文节注

- 两层编码 + 确定性最早开工解码；搜索/评测场景分离。
- Destroy：D1 随机；D3 关键路径；D4 高 CV；D5 机器载荷（P0 启用）。
- Repair：R4 SPT 偏置 + 局部重插；外加机器爬山。
- ALNS roulette + SA；权重按 segment 反应更新；**评测集不参与搜索**。
- LLM 仅作可选消融叙事；P0 表中未使用。

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

