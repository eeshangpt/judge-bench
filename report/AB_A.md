# A: cascade (MiniCheck accepts, LM takes the rest)

## [PRIMARY] ragtruth n=300, LM = Qwen_Qwen3-4B-Instruct-2507; Sonnet kappa 0.601, target 0.9x = 0.541

| quantity | out-of-fold point | nested 95% CI |
|---|---|---|
| A1 coverage | 0.213 | [0.027, 0.577] (n=1000) |
| selective kappa (accepted) | 0.527 | [-0.142, 0.567] (n=978) |
| selective risk (accepted) | 0.234 | [0.194, 0.562] (n=1000) |
| A2 cascade kappa (all items) | 0.313 | [0.204, 0.412] (n=1000) |
| cascade error | 0.343 | [0.293, 0.400] (n=1000) |
| LM-alone error | 0.347 | [0.297, 0.400] (n=1000) |
| cascade error minus LM-alone error | -0.003 | [-0.040, 0.030] (n=1000) |

Cost (prereg: (1 - coverage) x LM p50 latency): LM p50 8.46 s/item, coverage 0.213 -> 6.65 s/item expected vs 8.46 s/item LM alone.

Operating points (confusion rows tn/fp/fn/tp of the accepted subset, y=truth, pred=MiniCheck; prevalence = share faithful):

| point | n accepted | tn | fp | fn | tp | prev accepted | prev full | bacc (descr.) | selective kappa |
|---|---|---|---|---|---|---|---|---|---|
| out-of-fold pooled | 64 | 21 | 6 | 9 | 28 | 0.578 | 0.500 | 0.767 | 0.527 |
| deployable tau=0.446 (fit on all n, in-sample) | 63 | 21 | 5 | 9 | 28 | 0.587 | 0.500 | 0.782 | 0.552 |

Descriptive: AURC 0.286, E-AURC 0.201, random-confidence baseline (overall error) 0.380; MiniCheck alone kappa 0.240, LM alone kappa 0.307.

## [SECONDARY (exploratory)] ragtruth n=300, LM = microsoft_Phi-4-mini-instruct; Sonnet kappa 0.601, target 0.9x = 0.541

| quantity | out-of-fold point | nested 95% CI |
|---|---|---|
| A1 coverage | 0.213 | [0.027, 0.577] (n=1000) |
| selective kappa (accepted) | 0.527 | [-0.142, 0.567] (n=978) |
| selective risk (accepted) | 0.234 | [0.194, 0.562] (n=1000) |
| A2 cascade kappa (all items) | 0.193 | [0.047, 0.368] (n=1000) |
| cascade error | 0.403 | [0.313, 0.477] (n=1000) |
| LM-alone error | 0.430 | [0.377, 0.487] (n=1000) |
| cascade error minus LM-alone error | -0.027 | [-0.117, 0.020] (n=1000) |

Cost (prereg: (1 - coverage) x LM p50 latency): LM p50 7.06 s/item, coverage 0.213 -> 5.55 s/item expected vs 7.06 s/item LM alone.

Operating points (confusion rows tn/fp/fn/tp of the accepted subset, y=truth, pred=MiniCheck; prevalence = share faithful):

| point | n accepted | tn | fp | fn | tp | prev accepted | prev full | bacc (descr.) | selective kappa |
|---|---|---|---|---|---|---|---|---|---|
| out-of-fold pooled | 64 | 21 | 6 | 9 | 28 | 0.578 | 0.500 | 0.767 | 0.527 |
| deployable tau=0.446 (fit on all n, in-sample) | 63 | 21 | 5 | 9 | 28 | 0.587 | 0.500 | 0.782 | 0.552 |

Descriptive: AURC 0.286, E-AURC 0.201, random-confidence baseline (overall error) 0.380; MiniCheck alone kappa 0.240, LM alone kappa 0.140.

## [PRIMARY] wice n=172, LM = Qwen_Qwen3-4B-Instruct-2507; Sonnet kappa 0.548, target 0.9x = 0.493

| quantity | out-of-fold point | nested 95% CI |
|---|---|---|
| A1 coverage | 0.814 | [0.500, 0.994] (n=1000) |
| selective kappa (accepted) | 0.502 | [0.325, 0.550] (n=999) |
| selective risk (accepted) | 0.243 | [0.213, 0.326] (n=1000) |
| A2 cascade kappa (all items) | 0.488 | [0.377, 0.563] (n=1000) |
| cascade error | 0.256 | [0.215, 0.314] (n=1000) |
| LM-alone error | 0.244 | [0.180, 0.308] (n=1000) |
| cascade error minus LM-alone error | 0.012 | [-0.041, 0.081] (n=1000) |

Cost (prereg: (1 - coverage) x LM p50 latency): LM p50 18.79 s/item, coverage 0.814 -> 3.50 s/item expected vs 18.79 s/item LM alone.

Operating points (confusion rows tn/fp/fn/tp of the accepted subset, y=truth, pred=MiniCheck; prevalence = share faithful):

| point | n accepted | tn | fp | fn | tp | prev accepted | prev full | bacc (descr.) | selective kappa |
|---|---|---|---|---|---|---|---|---|---|
| out-of-fold pooled | 140 | 70 | 4 | 30 | 36 | 0.471 | 0.500 | 0.746 | 0.502 |
| deployable tau=0.387 (fit on all n, in-sample) | 135 | 67 | 4 | 29 | 35 | 0.474 | 0.500 | 0.745 | 0.500 |

Descriptive: AURC 0.173, E-AURC 0.123, random-confidence baseline (overall error) 0.297; MiniCheck alone kappa 0.407, LM alone kappa 0.512.

## [SECONDARY (exploratory)] wice n=172, LM = microsoft_Phi-4-mini-instruct; Sonnet kappa 0.548, target 0.9x = 0.493

| quantity | out-of-fold point | nested 95% CI |
|---|---|---|
| A1 coverage | 0.814 | [0.500, 0.994] (n=1000) |
| selective kappa (accepted) | 0.502 | [0.325, 0.550] (n=999) |
| selective risk (accepted) | 0.243 | [0.213, 0.326] (n=1000) |
| A2 cascade kappa (all items) | 0.523 | [0.365, 0.567] (n=1000) |
| cascade error | 0.238 | [0.215, 0.320] (n=1000) |
| LM-alone error | 0.256 | [0.192, 0.320] (n=1000) |
| cascade error minus LM-alone error | -0.017 | [-0.064, 0.076] (n=1000) |

Cost (prereg: (1 - coverage) x LM p50 latency): LM p50 15.48 s/item, coverage 0.814 -> 2.88 s/item expected vs 15.48 s/item LM alone.

Operating points (confusion rows tn/fp/fn/tp of the accepted subset, y=truth, pred=MiniCheck; prevalence = share faithful):

| point | n accepted | tn | fp | fn | tp | prev accepted | prev full | bacc (descr.) | selective kappa |
|---|---|---|---|---|---|---|---|---|---|
| out-of-fold pooled | 140 | 70 | 4 | 30 | 36 | 0.471 | 0.500 | 0.746 | 0.502 |
| deployable tau=0.387 (fit on all n, in-sample) | 135 | 67 | 4 | 29 | 35 | 0.474 | 0.500 | 0.745 | 0.500 |

Descriptive: AURC 0.173, E-AURC 0.123, random-confidence baseline (overall error) 0.297; MiniCheck alone kappa 0.407, LM alone kappa 0.488.

## Decision (prereg rules, primary LM = Qwen3-4B)

coverage {'ragtruth': np.float64(0.21333333333333335), 'wice': np.float64(0.813953488372093)}; nested lower bounds {'ragtruth': np.float64(0.02658333333333334), 'wice': np.float64(0.4998546511627907)}; A2 >= target {'ragtruth': np.False_, 'wice': np.False_}
**A verdict: ITERATE**
