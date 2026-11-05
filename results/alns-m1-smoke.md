# SA-ALNS M1 冒烟（vs MOR）

- 时间：2026-11-05（Asia/Shanghai）
- 协议对齐：`results/stochastic-fjsp-smoke.md`（3 例 SD3 10×5，评测切片 `[250:300]`，K_eval=50，seed=50）
- 搜索场景：realizations `[0:40]`（不与评测切片重叠）
- 方法：best-of-PDR 初解 + D1 `random_ops` + R4 `spt_biased` + 机器局部改进 + SA-ALNS

- `10x5+mix_001.fjs`: MOR E[Cmax]=757.04 → ALNS 650.12 (gap -14.12%), VaR95 882.35→778.65, iters=350, 11.8s
- `10x5+mix_002.fjs`: MOR E[Cmax]=674.42 → ALNS 651.84 (gap -3.35%), VaR95 770.40→746.25, iters=350, 11.5s
- `10x5+mix_003.fjs`: MOR E[Cmax]=700.94 → ALNS 619.82 (gap -11.57%), VaR95 809.20→690.75, iters=350, 12.7s

## 总表（评测期望 Cmax / VaR95）

| 实例 | MOR E[Cmax] | SA-ALNS E[Cmax] | gap% | MOR VaR95 | ALNS VaR95 | 迭代 | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10x5+mix_001.fjs | 757.04 | 650.12 | -14.12 | 882.35 | 778.65 | 350 | 11.8 |
| 10x5+mix_002.fjs | 674.42 | 651.84 | -3.35 | 770.40 | 746.25 | 350 | 11.5 |
| 10x5+mix_003.fjs | 700.94 | 619.82 | -11.57 | 809.20 | 690.75 | 350 | 12.7 |
| **均值** | **710.80** | **640.59** | **-9.88** | **820.65** | **738.55** | — | — |

## 与 PDR 冒烟报告对齐

| 参照 | 期望 Cmax 均值 |
|---|---:|
| 官方 env PDR 冒烟 MOR（`stochastic-fjsp-smoke.md`） | 686.97 |
| 本包 decoder 上 MOR | 710.80 |
| 本包 SA-ALNS M1 | 640.59 |

## 结论

- vs 本包 MOR：**PASS**（640.59 < 710.80）
- vs 冒烟 MOR 686.97：**PASS**（640.59 < 686.97）
- 总耗时 37.1s

```bash
cd /workspace/fjsp-combo-b && source .venv/bin/activate
PYTHONPATH=src python scripts/alns_m1_smoke.py
```
