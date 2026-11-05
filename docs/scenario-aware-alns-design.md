# 场景感知 ALNS/ILS 设计说明书（Stochastic FJSP）

对照基线：Smit et al., *Neural Combinatorial Optimization for Stochastic Flexible Job Shop Scheduling Problems*, AAAI 2025（SPM-DAN / CP-stoch / PDR）。  
冒烟基准：`/workspace/fjsp-combo-b/results/stochastic-fjsp-smoke.md`（MOR 期望 Cmax≈686.97 / VaR95≈793.85，SD3 10×5 前 3 例 × 50 场景）。  
实现骨架：`/workspace/fjsp-combo-b/src/sfjsp_alns/`。  
日期：2026-11-05（Asia/Shanghai）。

---

## 0. 目标与差异化

| 维度 | Smit AAAI 2025 | 本方法（场景感知 ALNS/ILS） |
|------|----------------|------------------------------|
| 决策形式 | 神经策略逐步派工 | 完整调度解上的邻域搜索 |
| 随机性处理 | SPM 注意力嵌入场景 | 显式场景采样评价 + 场景感知 destroy |
| 目标 | E[Cmax] 或 VaRα | 同；搜索中用 **搜索集**，报告用 **评测集** |
| LLM | 无 | **阶段 2 可选**：算子选择 / 修复偏好（承接 NS4S） |
| 可复现性 | 需预训练权重 | 无权重可跑；先无 LLM 强基线 |

投稿叙事：在相同随机协议上，用可解释的元启发式挑战纯 NCO；LLM 仅作消融增量，禁止「只有 LLM 才有效」。

---

## 1. 问题与数据约定（对齐官方仓）

### 1.1 确定性实例（`.fjs`）

- 路径示例：`baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/SD3/10x5+mix/*.fjs`
- 加载：`job_length: (J,)`, `op_pt: (N, M)`，`op_pt[i,j]=0` 表示工序 i 不能在机器 j 上加工
- 官方：`data_utils.load_data_from_single_file`

### 1.2 随机实现文件

- 路径：`data/stochastic_realizations/{variance}/{data_source}/{set}/{name}.fjs`
- 结构：首块为确定性实例；其后多个空行分隔的 realization（加工时间矩阵）
- 官方评测切片：`realizations[250 : 250 + num_eval]`（冒烟用 50；全文对齐论文时用 1000）
- `variance=random` + `variance_dist=lognormal` 时 **无** 额外 dist 子目录

### 1.3 内部数据结构

```text
Instance
  n_jobs, n_machines, n_ops
  job_length: ndarray[J]
  op_pt_nom: ndarray[N,M]          # 名义/中位加工时间
  compatible[i] = {m | op_pt[i,m] > 0}

ScenarioBundle
  op_pt_scen: ndarray[K,N,M]       # K 个场景
  role: "search" | "eval"

Solution（两层编码）
  machine_assign: ndarray[N]       # 每道工序选定机器（∈ compatible）
  op_sequence: list[N]             # 全局工序优先序（尊重同工件工序前后）
  # 解码：按序插入到所选机器甘特；同机用序列打破冲突 → Cmax(s)
```

解码必须确定性：相同 `(Solution, op_pt)` → 相同 Cmax。

---

## 2. 目标函数与场景协议

### 2.1 指标

对场景集 \(S=\{s_1,\ldots,s_K\}\)，解 \(x\)：

- **期望**：\(\hat{\mathbb{E}}[C_{\max}](x)=\frac{1}{K}\sum_k C_{\max}(x;s_k)\)
- **VaRα**：\(\widehat{\mathrm{VaR}}_{\alpha}(x)=\mathrm{Percentile}_{100\alpha}\{C_{\max}(x;s_k)\}\)（默认 α=0.95，与冒烟/论文一致）

搜索目标 `obj ∈ {mean, VaR}`；报告时两者都算。

### 2.2 搜索集 vs 评测集（防过拟合场景）

| 阶段 | K | 场景来源 | 用途 |
|------|---|----------|------|
| 搜索 | K_search=20～50 | 从 realizations 池 `[0:250)` 子采样或在线重采样 | ALNS 接受准则 |
| 精炼 | K_refine=100 | 固定子集 | 接受「准最优」后复核 |
| 报告 | K_eval=50（smoke）/1000（全文） | **固定** `[250:250+K_eval]`，与 PDR/Smit 对齐 | 论文表 |

规则：**禁止**用评测切片调参或选算子权重。

### 2.3 场景感知代价

每次邻域评价对 K_search 个场景解码 → 复杂度 O(K · decode)。加速：

1. **增量解码**（机器局部重排时只重算受影响机器）—— v1  
2. **公共随机数**：候选解共用同一搜索场景集  
3. **两级评价**：先 K_fast=5 筛掉明显更差，再全 K_search  
4. 预算：按「解码次数」或墙钟秒数封顶（与 CP-SAT 1h 对照时单独记）

---

## 3. 初始解

1. **PDR 种子**：FIFO / MOR / SPT / MWKR 在名义 `op_pt_nom` 上构造 `Solution`（复用冒烟动作语义）  
2. **多起点**：4 个 PDR + 2 个随机可行解 → 取搜索目标最优进 ALNS  
3. **可选 warm-start**：确定性 NS4S / DAN 解映射为 `(machine_assign, op_sequence)`（阶段 1.5）

冒烟对照：ALNS 报告期望 Cmax 应 **≤ MOR≈686.97**（同 3 例、同 50 评测场景）；否则实现有 bug。

---

## 4. 编码与邻域（Destroy / Repair）

### 4.1 Destroy 算子（权重自适应）

| ID | 名称 | 行为 | 场景感知？ |
|----|------|------|------------|
| D1 | `random_ops` | 随机移除 q% 工序的机器赋值（保留序） | 否 |
| D2 | `random_seq` | 随机打乱一段序列窗口 | 否 |
| D3 | `critical_path` | 在名义或最坏搜索场景上取关键路径工序 | **是** |
| D4 | `high_var_ops` | 按场景间加工时间 CV 或 Cmax 贡献方差选工序 | **是** |
| D5 | `machine_load` | 卸下载荷最高 / 场景载荷方差最大的机器上的工序 | **是** |
| D6 | `job_block` | 移除某工件全部工序（再插入） | 否 |

q 默认 U(0.1, 0.3)·N。D3–D5 为相对 Smit「隐式场景」的显式差异点。

### 4.2 Repair 算子

| ID | 名称 | 行为 |
|----|------|------|
| R1 | `greedy_mean` | 对被毁工序，枚举可行机+插入位，最小化搜索集 mean Cmax |
| R2 | `greedy_var` | 同上，目标 VaRα |
| R3 | `regret` | 经典 regret-k（k=2/3）插入 |
| R4 | `spt_biased` | 偏置最短加工时间机器，再局部插入 |
| R5 | `beam` | 束宽 B=3 的部分枚举（小实例） |

昂贵 repair（R1/R2）配合两级评价；大实例默认 R3/R4。

### 4.3 ILS 外层（可选与 ALNS 嵌套）

```text
x ← Init()
x* ← x
while budget:
  x' ← Perturb(x, strength)     # 强扰动 ≈ 大 q 的 D1+D6
  x' ← ALNS_local(x', iters_loc)
  if Accept_ILS(x', x): x ← x'
  if better(x, x*): x* ← x
```

阶段 0 可先实现 **纯 ALNS**；ILS 作外层重启。

---

## 5. ALNS 主循环与接受准则

```text
weights_D, weights_R ← 均匀
x ← best_init; x_best ← x
T ← T0
for it = 1..MaxIt or until time_limit:
  d ~ roulette(weights_D); r ~ roulette(weights_R)
  Ω ← d.destroy(x, q, scenarios_search)
  y ← r.repair(x, Ω, scenarios_search)   # 两级评价
  Δ ← obj(y) - obj(x)
  if Δ < 0 or rand() < exp(-Δ / T):      # SA 接受
     x ← y; update_scores(d,r, improvement|accept)
  if obj(x) < obj(x_best) on refine set:
     x_best ← x
  if it % segment == 0: update_weights(scores); T *= cooling
return x_best 再在 eval set 上报告 mean & VaR95
```

超参初值（10×5）：`MaxIt=2000`，`segment=50`，`T0=0.05·obj(x0)`，`cooling=0.999`，`reaction_factor=0.2`。

---

## 6. 阶段 2：LLM 辅助算子选择（可选，消融必备）

**不做**端到端生成完整调度（与 NS4S/Smit 边界清晰）。

| 模式 | 输入 | 输出 | 频率 |
|------|------|------|------|
| A 在线选择 | 特征：载荷、关键路径长度、场景方差、近 segment 算子得分 | 选 (D,R) 或调 q | 每 segment |
| B 离线进化 | 历史轨迹 + 算子描述 | 新 repair 偏好 / 权重先验 | 实验前 |
| C 禁用 | — | 经典 roulette | 主表强基线 |

特征必须数值化、可复现；LLM 失败则回退 roulette。消融：`ALNS` vs `ALNS+LLM-A`。

---

## 7. 实验计划（对标）

### 7.1 对照方法

| 方法 | 来源 | 备注 |
|------|------|------|
| FIFO/MOR/SPT/MWKR | 官方 `test_heuristic_stoch.py` / 冒烟脚本 | 已有数字 |
| CP-SAT（中位数） | 官方 Job_Shop… CP-SAT | 1h 上限单独记 |
| CP-stoch | 官方多场景 CP | 场景数按论文 |
| DAN / DAN-stoch / SPM-DAN | 官方；**权重缺失则表中 N/A 或自训后补** | 不虚构 |
| **SA-ALNS（本文）** | `src/sfjsp_alns` | 无 LLM |
| SA-ALNS+LLM | 阶段 2 | 消融 |

### 7.2 实例与指标

- 主：SD3 `10x5+mix` → `20x5` → `15x10`（先小后大）  
- 扩展：SD1；mk 加噪（与论文一致声明）  
- 指标：期望 Cmax、VaR95、相对 MOR gap%、评测墙钟、解码次数  
- 种子：≥5 个独立运行；报均值±标准差  
- 公平：同一 `K_eval` 切片；同一机器/线程声明

### 7.3 成功判据（实现级）

1. smoke 3 例：SA-ALNS 期望 Cmax **严格优于** 同协议 MOR  
2. 权重自适应曲线可复现；关掉 D3–D5 后性能下降（消融）  
3. 全量 10×5 上相对 MOR gap 稳定为负  
4. （可选）接近或超过公开可复现的 CP-stoch / 自训 SPM-DAN

---

## 8. 包结构与实现里程碑

```text
src/sfjsp_alns/
  __init__.py
  types.py           # Instance, ScenarioBundle, Solution, Metrics
  io/fjsp_loader.py  # 读 .fjs / realizations（对齐官方格式）
  decode.py          # Solution + op_pt → Cmax / 甘特
  evaluate.py        # mean / VaR over ScenarioBundle
  init_pdr.py        # PDR → Solution
  operators/destroy.py
  operators/repair.py
  alns.py            # 主循环
  ils.py             # 外层（stub）
  llm/selector.py    # 阶段 2 stub
  cli.py             # python -m sfjsp_alns.cli ...
tests/test_decode_smoke.py
```

| 里程碑 | 内容 | 验收 |
|--------|------|------|
| M0 | types + loader + decode + evaluate | 固定解在 50 场景上复现冒烟量级 |
| M1 | PDR init + D1/R4 + 简单 SA | 3 例优于 MOR |
| M2 | D3–D5 + R1/R3 + 自适应权重 | 消融成立 |
| M3 | CLI + CSV 出表 + 时限 | 可跑 10×5 全量 |
| M4 | LLM selector 接口 + 消融 | 可开关 |

---

## 9. 风险与边界

- 官方 SPM-DAN 权重缺失 → 表中如实 N/A，不以未训练模型冒充  
- `generate_random_instances.py` Windows 路径 bug → 继续用冒烟脚本 / 本包 loader  
- 场景过拟合 → 强制 search/eval 分离  
- 与组内 NS4S：本工作聚焦 **随机目标 + 场景算子**；确定性 LLM 邻域仅作 warm-start/阶段 2

---

## 10. 立刻可执行的 3 个编码任务

见仓库根 README 附录与本次回复「下一步」；对应 M0→M1。
