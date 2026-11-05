# K_eval held-out SA-ALNS

- kind: `external_sd3`
- comparable_to_paper_protocol: **True**
- instances: 20
- eval slice: `[250, 1250]` (K_eval=1000)
- search slice: `[0, 50]`
- seeds: [50, 51, 52]
- ALNS: max_iter=300, time_limit_s=40.0

Eval slice is held out from search. Synthetic kind uses shipped tiny fixtures and an in-process lognormal random-CoV draw; it is a wiring check, not an SD3 table. External kind reads --data-root and does not resample the bank.

| Method | E[Cmax] | VaR95 |
|---|---:|---:|
| FIFO | 662.0000 | 775.3275 |
| MOR | 655.5870 | 771.5925 |
| SPT | 828.7886 | 961.8367 |
| MWKR | 653.1490 | 766.2908 |
| SA-ALNS | 602.0756 | 694.0700 |

N/A — pretrained SPM-DAN/DAN-stoch weights are not in the public NCO-for-Stochastic-FJSP checkout; this harness does not invent a score.

