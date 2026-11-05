# P0 baselines comparison (MOR / SA-ALNS / CP-stoch-60s / SPM-DAN)

Protocol: 20× SD3 `10x5+mix`, eval `[250:350]` K=100, same decoder.

| Method | E[Cmax] | VaR95 | vs MOR (%) | notes |
|---|---:|---:|---:|---|
| MOR | 654.31 | 762.64 | +0.00 | PDR |
| **SA-ALNS** | **597.16** | **683.98** | **-8.73** | 300 iter / 40s, K_search=50 |
| CP-stoch (short) | 618.92 | 722.35 | -5.41 | official mean CP-SAT, **60s**, K_cp=10; **not** paper 3600s |
| SPM-DAN | N/A | N/A | — | blocked: no pretrained FJSP weights; short CPU train unfair / deps missing |

SA-ALNS beats CP-stoch-60s on **18/20** instances (seed-mean E vs single CP run).

## Per-instance E[Cmax]

| Instance | MOR | SA-ALNS | CP-stoch-60s | ALNS−CP |
|---|---:|---:|---:|---:|
| 10x5+mix_001 | 741.66 | 654.59 | 704.25 | -49.66 |
| 10x5+mix_002 | 679.74 | 641.65 | 642.16 | -0.51 |
| 10x5+mix_003 | 665.37 | 616.39 | 628.89 | -12.50 |
| 10x5+mix_004 | 635.13 | 571.57 | 564.78 | 6.79 |
| 10x5+mix_005 | 583.81 | 551.30 | 576.62 | -25.32 |
| 10x5+mix_006 | 612.00 | 562.26 | 600.82 | -38.56 |
| 10x5+mix_007 | 558.59 | 517.34 | 524.15 | -6.81 |
| 10x5+mix_008 | 843.46 | 710.47 | 726.16 | -15.69 |
| 10x5+mix_009 | 717.59 | 661.08 | 689.13 | -28.05 |
| 10x5+mix_010 | 649.39 | 592.84 | 629.11 | -36.27 |
| 10x5+mix_011 | 575.36 | 524.18 | 554.12 | -29.94 |
| 10x5+mix_012 | 635.18 | 609.64 | 607.80 | 1.84 |
| 10x5+mix_013 | 676.73 | 638.25 | 664.15 | -25.90 |
| 10x5+mix_014 | 697.12 | 636.77 | 658.20 | -21.43 |
| 10x5+mix_015 | 643.11 | 562.24 | 569.73 | -7.49 |
| 10x5+mix_016 | 650.81 | 596.56 | 606.30 | -9.74 |
| 10x5+mix_017 | 639.57 | 583.44 | 622.35 | -38.91 |
| 10x5+mix_018 | 641.25 | 579.38 | 620.29 | -40.91 |
| 10x5+mix_019 | 636.23 | 583.36 | 619.24 | -35.88 |
| 10x5+mix_020 | 604.09 | 549.96 | 570.08 | -20.12 |

**Caveat:** CP search objective on K=10 was macro 573.07 (optimistic vs held-out 618.92). Paper CP uses up to 3600s and often VaR; these numbers are a **matched-protocol short-budget** baseline only.

