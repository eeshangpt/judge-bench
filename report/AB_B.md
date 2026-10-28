# B: rank generators (RAGTruth QA test, 900 items)

## Qwen_Qwen3-4B-Instruct-2507 (PRIMARY)

| generator | human halluc. rate H | judged unfaithful rate U |
|---|---|---|
| gpt-3.5-turbo-0613 | 0.033 | 0.167 |
| gpt-4-0613 | 0.007 | 0.073 |
| llama-2-13b-chat | 0.240 | 0.240 |
| llama-2-70b-chat | 0.233 | 0.200 |
| llama-2-7b-chat | 0.347 | 0.340 |
| mistral-7B-instruct | 0.207 | 0.260 |

Kendall tau-b = 0.733; bootstrap 95% CI (undefined counted 0) [0.414, 1.000] (n=1000); CI with undefined dropped [0.414, 1.000] (n=1000); undefined resamples 0; exact permutation p = 0.056 (descriptive)
Tier order correct (U(7b) > max U(Mid), min U(Mid) > max U(Low)): True. Spearman rho 0.829; tau-b with mean(1-p_yes) score 0.733.
Item level on 900 (natural mix, prevalence faithful 0.822): kappa 0.161, bacc 0.587.
**PRIMARY verdict by rule: SCALE (pilot; worth a run on fresh data)**

## minicheck (exploratory)

| generator | human halluc. rate H | judged unfaithful rate U |
|---|---|---|
| gpt-3.5-turbo-0613 | 0.033 | 0.267 |
| gpt-4-0613 | 0.007 | 0.267 |
| llama-2-13b-chat | 0.240 | 0.340 |
| llama-2-70b-chat | 0.233 | 0.360 |
| llama-2-7b-chat | 0.347 | 0.460 |
| mistral-7B-instruct | 0.207 | 0.427 |

Kendall tau-b = 0.552; bootstrap 95% CI (undefined counted 0) [0.276, 0.966] (n=1000); CI with undefined dropped [0.276, 0.966] (n=1000); undefined resamples 0; exact permutation p = 0.189 (descriptive)
Tier order correct (U(7b) > max U(Mid), min U(Mid) > max U(Low)): True. Spearman rho 0.754; tau-b with mean(1-p_yes) score 0.600.
Item level on 900 (natural mix, prevalence faithful 0.822): kappa 0.129, bacc 0.589.
**exploratory verdict by rule: SCALE (pilot; worth a run on fresh data)**

## microsoft_Phi-4-mini-instruct (exploratory)

| generator | human halluc. rate H | judged unfaithful rate U |
|---|---|---|
| gpt-3.5-turbo-0613 | 0.033 | 0.307 |
| gpt-4-0613 | 0.007 | 0.373 |
| llama-2-13b-chat | 0.240 | 0.500 |
| llama-2-70b-chat | 0.233 | 0.453 |
| llama-2-7b-chat | 0.347 | 0.500 |
| mistral-7B-instruct | 0.207 | 0.380 |

Kendall tau-b = 0.828; bootstrap 95% CI (undefined counted 0) [0.200, 0.867] (n=1000); CI with undefined dropped [0.200, 0.867] (n=1000); undefined resamples 0; exact permutation p = 0.028 (descriptive)
Tier order correct (U(7b) > max U(Mid), min U(Mid) > max U(Low)): False. Spearman rho 0.928; tau-b with mean(1-p_yes) score 0.867.
Item level on 900 (natural mix, prevalence faithful 0.822): kappa 0.114, bacc 0.587.
**exploratory verdict by rule: KILL**

## B verdict (primary judge Qwen3-4B): SCALE (pilot; worth a run on fresh data)
