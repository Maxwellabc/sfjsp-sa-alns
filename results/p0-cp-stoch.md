# P0 CP-stoch (short budget) vs SA-ALNS protocol

- Instances: 20 × SD3 `10x5+mix`
- CP: OR-Tools CP-SAT stochastic **mean**, K_cp=10, time_limit=60s
- Eval: re-decode CP assignment+sequence on scenarios [250:350] (same decoder as P0)
- Feasible: 20/20
- Macro E[Cmax] (eval): 618.9164999999999
- Macro VaR95 (eval): 722.3475
- Macro CP search obj: 573.065
- Mean solve wall: 60.04965s; total 1214.8s

| Instance | status | CP search obj | eval E[Cmax] | VaR95 | wall_s |
|---|---|---:|---:|---:|---:|
| 10x5+mix_001 | FEASIBLE | 634.8000000000001 | 704.25 | 821.8 | 60.05 |
| 10x5+mix_002 | FEASIBLE | 609.1 | 642.16 | 723.8499999999999 | 60.044 |
| 10x5+mix_003 | FEASIBLE | 605.6 | 628.89 | 703.2 | 60.064 |
| 10x5+mix_004 | FEASIBLE | 540.6 | 564.78 | 676.4 | 60.042 |
| 10x5+mix_005 | FEASIBLE | 524.9 | 576.62 | 668.4499999999999 | 60.042 |
| 10x5+mix_006 | FEASIBLE | 534.0 | 600.82 | 715.05 | 60.051 |
| 10x5+mix_007 | FEASIBLE | 486.09999999999997 | 524.15 | 606.05 | 60.045 |
| 10x5+mix_008 | FEASIBLE | 679.1 | 726.16 | 847.55 | 60.044 |
| 10x5+mix_009 | FEASIBLE | 657.0 | 689.13 | 805.4499999999999 | 60.044 |
| 10x5+mix_010 | FEASIBLE | 583.7 | 629.11 | 748.0 | 60.052 |
| 10x5+mix_011 | FEASIBLE | 513.0000000000001 | 554.12 | 644.1 | 60.049 |
| 10x5+mix_012 | FEASIBLE | 570.4 | 607.8 | 694.55 | 60.044 |
| 10x5+mix_013 | FEASIBLE | 602.5 | 664.15 | 797.05 | 60.055 |
| 10x5+mix_014 | FEASIBLE | 604.8000000000001 | 658.2 | 749.05 | 60.068 |
| 10x5+mix_015 | FEASIBLE | 544.4000000000001 | 569.73 | 667.05 | 60.046 |
| 10x5+mix_016 | FEASIBLE | 579.0 | 606.3 | 695.55 | 60.041 |
| 10x5+mix_017 | FEASIBLE | 558.2 | 622.35 | 734.25 | 60.059 |
| 10x5+mix_018 | FEASIBLE | 533.8000000000001 | 620.29 | 770.3 | 60.052 |
| 10x5+mix_019 | FEASIBLE | 564.1 | 619.24 | 735.8499999999999 | 60.045 |
| 10x5+mix_020 | FEASIBLE | 536.1999999999999 | 570.08 | 643.4 | 60.056 |

## SPM-DAN
N/A — blocked: no pretrained SPM-DAN/DAN-stoch checkpoints in NCO-for-Stochastic-FJSP (GitHub releases empty for FJSP weights); short CPU train unfair vs paper and missing deps (gym, torch_geometric, einops) at probe time.

