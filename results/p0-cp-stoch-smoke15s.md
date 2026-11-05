# P0 CP-stoch (short budget 15s) vs SA-ALNS protocol

- Instances: 1 × SD3 `10x5+mix`
- CP: OR-Tools CP-SAT stochastic **mean**, K_cp=10, time_limit=15s, workers=2
- Eval: re-decode CP assignment+sequence on scenarios [250:350] (same decoder as P0)
- Feasible: 1/1
- Macro E[Cmax] (eval): 707.31
- Macro VaR95 (eval): 876.3
- Macro CP search obj: 663.8000000000001
- Mean solve wall: 15.012s; total 15.6s
- Label: **short-budget**; do not equate to paper 3600s

| Instance | status | CP search obj | eval E[Cmax] | VaR95 | wall_s |
|---|---|---:|---:|---:|---:|
| 10x5+mix_001 | FEASIBLE | 663.8000000000001 | 707.31 | 876.3 | 15.012 |

## SPM-DAN
N/A — blocked: no pretrained SPM-DAN/DAN-stoch checkpoints in NCO-for-Stochastic-FJSP (GitHub releases empty for FJSP weights); short CPU train unfair vs paper and missing deps (gym, torch_geometric, einops) at probe time.

