# K_eval held-out SA-ALNS

- kind: `external_sd3`
- comparable_to_paper_protocol: **True**
- instances: 1
- eval slice: `[250, 1250]` (K_eval=1000)
- search slice: `[0, 50]`
- seeds: [50]
- ALNS: max_iter=20, time_limit_s=15.0

Eval slice is held out from search. Synthetic kind uses shipped tiny fixtures and an in-process lognormal random-CoV draw; it is a wiring check, not an SD3 table. External kind reads --data-root and does not resample the bank.

| Method | E[Cmax] | VaR95 |
|---|---:|---:|
| FIFO | 777.8950 | 911.0500 |
| MOR | 770.5440 | 921.0000 |
| SPT | 819.8560 | 942.0000 |
| MWKR | 829.8790 | 988.0500 |
| SA-ALNS | 696.0890 | 821.0500 |

N/A — pretrained SPM-DAN/DAN-stoch weights are not in the public NCO-for-Stochastic-FJSP checkout; this harness does not invent a score.

