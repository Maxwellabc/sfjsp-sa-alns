# Stochastic FJSP（Smit et al. AAAI 2025）小规模冒烟评测

- 时间：2026-11-05（Asia/Shanghai）
- 工作区：`/workspace/fjsp-combo-b/baselines/NCO-for-Stochastic-FJSP/FJSP-DRL`
- Python：3.13.5；torch=2.15.0+cpu；numpy=2.5.4；CUDA=False
- 策略：**不训练**；官方 FJSP-DRL 仓未附带预训练权重，故冒烟跑 **PDR 启发式**（论文基线对照：FIFO/MOR/SPT/MWKR），验证场景生成→确定性派工→随机场景回放→期望 Cmax / VaR95 全链路。

## 配置

| 项 | 值 |
|---|---|
| data_source | `smoke_SD3`（自 SD3/10x5+mix 取前 3 例） |
| instances | 3（001–003） |
| variance | `random` / `lognormal` |
| realization samples | 350（评测切片 `[250:300]`） |
| num_eval_realizations | 50 |
| seed | 50 |
| methods | FIFO, MOR, SPT, MWKR |

- 场景文件目录：`/workspace/fjsp-combo-b/baselines/NCO-for-Stochastic-FJSP/FJSP-DRL/data/stochastic_realizations/random/smoke_SD3/10x5+mix`

- 加载实例数：3（集名 `10x5+mix`）

## 结果（单次 seed；实例级均值）

| 方法 | 确定性 Cmax（均值） | 期望 Cmax（场景均值） | VaR95 Cmax | 派工耗时 s（均值） |
|---|---:|---:|---:|---:|
| FIFO | 610.33 | 706.75 | 826.07 | 0.0264 |
| MOR | 602.67 | 686.97 | 793.85 | 0.0303 |
| SPT | 689.00 | 779.01 | 891.83 | 0.0262 |
| MWKR | 596.67 | 699.61 | 799.93 | 0.0265 |

### 分实例明细（期望 Cmax）

| 实例 | FIFO | MOR | SPT | MWKR |
|---|---:|---:|---:|---:|
| 10x5+mix_001.fjs | 702.04 | 699.60 | 732.80 | 714.60 |
| 10x5+mix_002.fjs | 749.38 | 663.78 | 855.52 | 697.70 |
| 10x5+mix_003.fjs | 668.82 | 697.54 | 748.70 | 686.54 |

### VaR95 分实例

| 实例 | FIFO | MOR | SPT | MWKR |
|---|---:|---:|---:|---:|
| 10x5+mix_001.fjs | 860.10 | 824.80 | 831.20 | 807.20 |
| 10x5+mix_002.fjs | 854.25 | 745.90 | 982.65 | 788.85 |
| 10x5+mix_003.fjs | 763.85 | 810.85 | 861.65 | 803.75 |

## 命令与环境

```bash
cd /workspace/fjsp-combo-b
source .venv/bin/activate
python scripts/sfjsp_smoke_eval.py
```

## 失败与修复

| 问题 | 处理 |
|---|---|
| `requirements.txt` 含 `torch==2.3.0+cu121`、`pywin32`、`gurobipy` | 冒烟用 CPU torch + 精简依赖（numpy/scipy/tqdm/pandas），未装 Windows/Gurobi 包 |
| 官方仓无 FJSP-DRL 预训练权重；评测脚本默认指向未提供的 `test_model` | 改跑论文中的 PDR 对照链路，证明场景协议与指标口径 |
| `generate_random_instances.py` 用 `split('\\')`（Windows） | 冒烟脚本用 `Path.stem` 写 Linux 路径 |
| 全量 100 例 × 5000 场景过大 | 缩至 3 例 × 350 场景，评测 50 场景 |
| 无 SPM-DAN 数值 | **非失败**：需作者权重或自行训练；下一步再考虑短训/索取权重 |

## 结论

- **冒烟 PASS**：PDR 全链路在 1.2s 内跑通，输出确定性 Cmax / 期望 Cmax / VaR95。
- 数值为 smoke 子集，**不可**直接与论文全文表对比；协议对齐后可扩到 SD3 全量 + 作者模型。
- 原始数组：`/workspace/fjsp-combo-b/results/stochastic-fjsp-smoke/`

