# Local small judges for RAG faithfulness: job 03 analysis (issue #146)

Date: 2026-10-06. All numbers are from `report/results.csv`, computed by `analyze.py` from `preds/` and `data/`, or read from `refs/2609.39229.txt` (RAIM). Bootstrap: 1,000 item resamples, seed 7, 95% percentile intervals. The tuned threshold maximises kappa on the 50-item calib split (grid 0.01-0.99, ties go to the value nearest 0.5) and is then fixed for the test sample. The CIs do not re-tune it.

## TL;DR

**Proposal: ITERATE.** Neither kill nor scale fires, under any reading of "both sets". The hypothesis (a local judge at or above 0.9x Sonnet kappa AND 1 s/item or less) is not met by any single model. The two conditions split across models:

- MiniCheck-FT5 meets the latency target (p50 0.18-0.98 s) but is far below 0.9x on RAGTruth.
- Qwen3-4B reaches 0.93x on WiCE but runs at 4.8-18.8 s/item p50.

Overall confidence in the proposal: **medium-high (70%)**.

## Rule application (issue 146)

"Both sets" in the issue means RAGTruth plus LLM-AggreFact. I took RAGTruth and WiCE as the two primary sets (RAIM core-grounded, both RAG-like) and checked the result on all five. Ratio = local kappa / RAIM Sonnet kappa. "Best" is the best of the three local models per set.

| set | RAIM Sonnet κ | best local @0.5 (ratio) | best local @tuned thr (ratio) |
|---|---|---|---|
| RAGTruth | 0.601 | Qwen3-4B 0.307 (0.51x) | Qwen3-4B 0.467 (0.78x), thr 0.95 |
| WiCE | 0.548 | Qwen3-4B 0.512 (0.93x) | Qwen3-4B 0.512 (0.93x) |
| XSum | 0.418 | MiniCheck 0.560 (1.34x) | MiniCheck 0.453 (1.08x) |
| CNN | 0.469 | MiniCheck 0.375 (0.80x) | Phi-4-mini 0.406 (0.87x) |
| ExpertQA | 0.165 | Qwen3-4B 0.227 (1.37x) | Phi-4-mini 0.273 (1.66x), thr 0.95 |

Ratio 95% bootstrap CIs for the headline cells (the CI resamples our items only; RAIM's Sonnet kappa is treated as a fixed constant, so the true uncertainty is wider):

| cell | ratio [95% CI] |
|---|---|
| RAGTruth, Qwen3-4B @0.5 | 0.51 [0.35, 0.67] |
| RAGTruth, Qwen3-4B tuned | 0.78 [0.60, 0.94] |
| WiCE, Qwen3-4B (both) | 0.93 [0.71, 1.15] |
| XSum, MiniCheck @0.5 | 1.34 [1.12, 1.55] |
| CNN, MiniCheck @0.5 | 0.80 [0.33, 1.26] |
| ExpertQA, Qwen3-4B @0.5 | 1.37 [0.76, 2.03] |

- **Kill** (best below 0.8x on both): not met. WiCE, XSum and ExpertQA are at or above 0.93x.
- **Scale** (at or above 0.9x on both): not met. RAGTruth is 0.51x at 0.5 and 0.78x tuned, and CNN is below 0.9x at 0.5.
- **Iterate** is therefore the outcome. It holds with the default threshold, the tuned threshold, the primary pair, and all five sets.
- Latency: no model satisfies both 0.9x kappa and 1 s/item or less on RAGTruth or WiCE. Only MiniCheck is at or under 1 s p50 (RAGTruth 0.29, WiCE 0.98), and its RAGTruth kappa ratio is 0.40-0.50. So the full hypothesis, including latency, fails for the causal LMs by about 10-20x.

## Results (test samples; full table in `report/results.csv`)

κ = Cohen's kappa. Brackets are 95% bootstrap CIs. T=0.5 is the default threshold. TUNED is the calib-tuned threshold, shown in parentheses.

| model | set | n | κ @T=0.5 | bacc @T=0.5 | κ TUNED (thr) | bacc TUNED | κ ratio @0.5 / TUNED | ECE | p50 / p95 s | peak VRAM MB |
|---|---|---|---|---|---|---|---|---|---|---|
| MiniCheck-FT5 | RAGTruth | 300 | 0.240 [0.133,0.348] | 0.620 | 0.300 (0.86) | 0.650 | 0.40 / 0.50 | 0.215 | 0.29 / 0.47 | 3363 |
| MiniCheck-FT5 | WiCE | 172 | 0.407 [0.282,0.530] | 0.703 | 0.442 (0.40) | 0.721 | 0.74 / 0.81 | 0.224 | 0.98 / 3.79 | 3941 |
| MiniCheck-FT5 | XSum | 300 | 0.560 [0.468,0.648] | 0.780 | 0.453 (0.90) | 0.727 | 1.34 / 1.08 | 0.154 | 0.29 / 0.80 | 3294 |
| MiniCheck-FT5 | CNN | 64 | 0.375 [0.153,0.593] | 0.688 | 0.344 (0.85) | 0.672 | 0.80 / 0.73 | 0.241 | 0.59 / 1.35 | 3311 |
| MiniCheck-FT5 | ExpertQA | 300 | 0.200 [0.090,0.307] | 0.600 | 0.060 (0.96) | 0.530 | 1.21 / 0.36 | 0.323 | 0.18 / 1.62 | 3489 |
| Qwen3-4B-Instruct-2507 (4-bit) | RAGTruth | 300 | 0.307 [0.212,0.405] | 0.653 | 0.467 (0.95) | 0.733 | 0.51 / 0.78 | 0.298 | 8.46 / 11.86 | 2679 |
| Qwen3-4B | WiCE | 172 | 0.512 [0.389,0.628] | 0.756 | 0.512 (0.50) | 0.756 | 0.93 / 0.93 | 0.242 | 18.79 / 54.54 | 2873 |
| Qwen3-4B | XSum | 300 | 0.380 [0.278,0.479] | 0.690 | 0.380 (0.50) | 0.690 | 0.91 / 0.91 | 0.296 | 6.56 / 14.52 | 2747 |
| Qwen3-4B | CNN | 64 | 0.312 [0.158,0.489] | 0.656 | 0.312 (0.50) | 0.656 | 0.67 / 0.67 | 0.362 | 11.46 / 24.40 | 2743 |
| Qwen3-4B | ExpertQA | 300 | 0.227 [0.125,0.335] | 0.613 | 0.233 (0.61) | 0.617 | 1.37 / 1.41 | 0.377 | 4.75 / 31.52 | 2873 |
| Phi-4-mini-instruct (4-bit) | RAGTruth | 300 | 0.140 [0.018,0.247] | 0.570 | 0.073 (0.65) | 0.537 | 0.23 / 0.12 | 0.141 | 7.06 / 9.85 | 2929 |
| Phi-4-mini | WiCE | 172 | 0.488 [0.359,0.617] | 0.744 | 0.442 (0.86) | 0.721 | 0.89 / 0.81 | 0.158 | 15.48 / 46.89 | 3130 |
| Phi-4-mini | XSum | 300 | 0.373 [0.272,0.477] | 0.687 | 0.360 (0.30) | 0.680 | 0.89 / 0.86 | 0.220 | 5.67 / 12.51 | 2994 |
| Phi-4-mini | CNN | 64 | 0.031 [0.000,0.106] | 0.516 | 0.406 (0.97) | 0.703 | 0.07 / 0.87 | 0.444 | 9.83 / 21.11 | 2990 |
| Phi-4-mini | ExpertQA | 300 | 0.173 [0.080,0.271] | 0.587 | 0.273 (0.95) | 0.637 | 1.05 / 1.66 | 0.319 | 4.17 / 27.40 | 3130 |

Latency is per item, wall clock on the GTX 1660 Ti, from the `samples` split. Peak VRAM is the max of `torch.cuda.max_memory_allocated` over items. Quirks:

- MiniCheck runs in fp32.
- The causal LMs run in 4-bit NF4 with a 4096-token prompt cap.
- Truncated items: WiCE 15 for both causal models, ExpertQA 8 for both (`truncated` flag in preds).

## RAIM comparison (κ on the same benchmarks; RAIM Table 1 and Table 12)

| set | RAIM Sonnet | RAIM stacked panel (10 × 4-9B) | RAIM MiniCheck-FT5 | our MiniCheck @0.5 | our best causal @0.5 |
|---|---|---|---|---|---|
| RAGTruth | 0.601 | 0.409 | 0.262 | 0.240 | 0.307 (Qwen) |
| WiCE | 0.548 | 0.540 | 0.459 | 0.407 | 0.512 (Qwen) |
| XSum | 0.418 | 0.437 | 0.484 | 0.560 | 0.380 (Qwen) |
| CNN | 0.469 | 0.349 | 0.349 | 0.375 | 0.312 (Qwen) |
| ExpertQA | 0.165 | 0.235 | 0.174 | 0.200 | 0.227 (Qwen) |

Our MiniCheck's κ agrees with RAIM's MiniCheck within our 95% CI on all 5 sets (largest gap XSum, 0.560 vs 0.484). This is a loose check (wide CIs, so it would not detect a small bias) and it covers only the data, metric and MiniCheck scoring path. It does not validate the causal-LM prompt, tokenisation or logprob path used for Qwen3-4B and Phi-4-mini. RAIM's panel and Sonnet are both better than any of our single local judges on RAGTruth, which is consistent with RAIM's finding that one strong judge dominates there.

### Where our samples differ from RAIM's, so the comparison is NOT like for like

1. **Subsampling.** Ours is n=300 per set (RAGTruth, XSum, ExpertQA) against RAIM's 1362, 546 and 1462. WiCE is 172 vs 222 and CNN is 64 vs 114, because 50 items per set were held out for calibration. Our CIs are therefore wide (CNN ±0.22).
2. **RAGTruth pairing.** RAIM keeps matched clean/hallucinated pairs per query (681 pairs). We built the same 681 pairs but then sampled items without regard to pairing (`build_data.py` shuffles pos/neg independently), so our sample generally does not keep the pairs. The class balance is exact (150/150), the construction is not identical.
3. **Sonnet figures are RAIM's, not our rerun.** We never ran Sonnet (no paid API), so the ratios use RAIM's numbers on RAIM's full samples against our subsamples.
4. **Prompt.** RAIM asks for a reasoning sentence plus a VERDICT line. We read a single-token Yes/No logprob with no reasoning, which likely disadvantages the causal LMs.
5. **Quantisation and context.** Our causal models are 4-bit NF4 with a 4096-token prompt cap (head-kept context truncation). RAIM used bf16 members with an 8192-token cap, so some of our WiCE and ExpertQA items were truncated.
6. **Models.** RAIM's members are 4-9B. Ours are Qwen3-4B-Instruct-2507 and Phi-4-mini-instruct, which are not in RAIM's panel. Qwen2.5-3B and SmolLM3 were cut for time.
7. **MiniCheck.** RAIM used the official package at its documented operating point. We reimplemented the scoring from `refs/minicheck_inference.py` at the 0.5 default.
8. **Data source.** LLM-AggreFact is gated, so we used the mirror NinaCalvi/llm_aggrefact_pre_aug9 (test split). The pool sizes matched RAIM's (identical balanced n), but this is not confirmed to be the same rows.
9. **Label polarity.** Ours is 1 = supported, which is inverted vs RAIM. Kappa and balanced accuracy are symmetric under this, so it has no effect on the numbers.

## Findings with confidence

| # | Finding | Confidence |
|---|---|---|
| F1 | Proposal is ITERATE: neither kill (WiCE/XSum/ExpertQA ≥0.9x) nor scale (RAGTruth is 0.51x at 0.5 and 0.78x tuned) fires, under any reading of "both sets". | high (85%) |
| F2 | No local judge meets 0.9x kappa and 1 s/item or less together. Causal LMs are 4-19 s p50, about 10-20x too slow. This is a consequence of the 4-bit bitsandbytes path on a 1660 Ti (no tensor cores, fp16 compute), not necessarily of the models. | high (85%) for speed measured; medium (60%) for attribution to hardware/quantisation path |
| F3 | RAGTruth is the weak set: every local judge is below 0.8x Sonnet at 0.5, and all but Qwen tuned are far below. | medium-high (80%) |
| F4 | On WiCE, Qwen3-4B (0.93x) and Phi-4-mini (0.89x) are close to Sonnet in point estimate. The CIs (0.39-0.63 for κ, ratios roughly 0.7-1.15) cannot separate them from either 0.8x or 1.0x. | low-medium (55%). BELOW THRESHOLD, not used for the proposal. |
| F5 | On XSum, CNN and ExpertQA, local judges match or beat RAIM Sonnet's low baselines (XSum 0.418, ExpertQA 0.165). This is a weak bar (ExpertQA Sonnet κ is 0.165, near noise), so the "≥1.2x" ratios there are not evidence of quality. | medium-high (70%) |
| F6 | Calibration-tuned thresholds help or hurt unpredictably. Qwen RAGTruth goes 0.307→0.467 (thr 0.95). Phi CNN goes 0.031→0.406 (thr 0.97, because Phi predicted "yes" on 98% of items at 0.5). MiniCheck ExpertQA goes 0.200→0.060. A 50-item calib split is too small to trust (threshold-tuning gain on RAGTruth for Qwen). | low-medium (50%). BELOW THRESHOLD. The 50-item split is itself the limiting factor. |
| F7 | All judges are poorly calibrated: ECE 0.14-0.44 (10 bins). Phi-4-mini is best on RAGTruth (0.141) and worst on CNN (0.444). MiniCheck is lowest on XSum (0.154). | medium-high (80%) |
| F8 | Our MiniCheck κ agrees with RAIM's MiniCheck κ within our CIs on all 5 sets (wide CIs, so this is a loose check). The data and metric plumbing and the MiniCheck path are credible; the causal-LM path is not validated by this check. | medium-high (70%) |
| F9 | Peak VRAM is 2.7-3.9 GB for every model (MiniCheck 3.3-3.9 GB in fp32). All fit the 6 GB card with headroom. | high (90%) |
| F10 | CNN (n=64) rankings and ratios are not informative (CI half-width about 0.2). | low (40%). BELOW THRESHOLD. |

**Overall confidence: medium-high (70%).** Rule: overall = the lowest confidence among the findings the proposal rests on (F1, F2, F3, F5, F7, F8, F9), which is 70% (F5, F8). F1 (85%) rests on arithmetic over our own numbers plus RAIM's published Sonnet kappa; the sample-construction caveats are carried by F5 and F8. F4, F6 and F10 were not used. (An earlier draft said 75%; revised after the audit because 75% exceeded two supporting findings.)

## Metric glossary (standard terms)

- **Cohen's kappa (κ)**: chance-corrected agreement between judge and gold labels. κ = (p_o - p_e) / (1 - p_e), p_o = observed agreement, p_e = sum over classes of P(gold=c) x P(pred=c). Cohen (1960), Educational and Psychological Measurement 20(1).
- **Balanced accuracy (bacc)**: mean of per-class recall, (TPR + TNR) / 2. Brodersen et al. (2010), ICPR.
- **Expected calibration error (ECE)**: sum over bins b of (|B_b|/n) x |acc(B_b) - conf(B_b)|; here 10 equal-width bins on p_yes (P(supported)), acc = gold-positive rate, conf = mean p_yes in the bin. Guo et al. (2017), ICML, "On Calibration of Modern Neural Networks"; Naeini et al. (2015), AAAI.
- **p_yes**: softmax over the Yes/No token logits at the answer position, P(Yes) (judge.py). MiniCheck uses its own supported-probability output.
- **Bootstrap percentile CI**: 1,000 resamples of items with replacement, 2.5th and 97.5th percentiles. Efron and Tibshirani (1993), An Introduction to the Bootstrap.
- **Latency p50 / p95**: 50th / 95th percentile of per-item wall-clock time.
- **Ratio**: local κ / RAIM Sonnet κ. This is a convenience ratio defined by the issue's hypothesis, not a standard metric; its CI is in the table above.
- **RAIM** (refs/2609.39229.txt): Sonnet and stacked-panel κ from Table 1, MiniCheck (Flan-T5-L column) κ from Table 12. All values were re-checked against those tables.

## Proposal

**ITERATE.** Concrete next steps, all $0 and local:

1. Run on the full RAGTruth 681 pairs and all of RAIM's items where compute allows. MiniCheck is cheap enough for the full sets at about 0.3 s/item.
2. Fix speed before more models. The 4-bit bnb path costs 4-19 s per item. Try fp16 with a shorter cap, a GGUF/llama.cpp build, or chunked prompts, and re-test the 1 s/item claim.
3. Test whether reasoning-style prompts (RAIM's) recover RAGTruth for Qwen3-4B, since the Yes/No-only variant is a deviation.
4. Larger calibration (200+ items) or the PPI route from the kill branch, since the 50-item tuning is unreliable (F6).
5. Add the deferred models (Qwen2.5-3B, SmolLM3) only after step 2.

## Learnings

**What worked**

- The resumable harness: `judge.py` skipped finished ids, so inference survived a machine shutdown with no lost work.
- MiniCheck reproduced RAIM's MiniCheck numbers within our (wide) CIs, which supports the data and metric plumbing (not the causal-LM path).
- Single-token Yes/No logprobs gave usable p_yes and an ECE for free.

**What failed**

- Throughput: 4-bit causal LMs on a 1660 Ti ran at 4-19 s/item, so latency was never close to the 1 s hypothesis. It also cut the model list from 5 to 3.
- 50-item calibration is too small to tune thresholds (F6).
- The setup job ended without a `JOB:` line and stalled the queue for hours.
- Phi-4-mini on CNN with the default threshold predicted "yes" almost always (kappa 0.03).

**Change next time**

- Smoke-test throughput before choosing the model list.
- Make the setup job's last-line contract explicit.
- Pre-register which sets count as "both sets" in the kill/scale rule (the issue was ambiguous, so I stated my reading above).
- Size samples for the CI target, and keep RAGTruth pairs.
- Run one Sonnet-class reference inside the same harness if budget ever allows, so ratios are not against another paper's samples.
