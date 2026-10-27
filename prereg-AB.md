# Pre-registration A+B, issue #146 (local small judges for RAG faithfulness)

Status: DRAFT v1, 2026-10-28. Not frozen: owner OK on hypotheses and kill criteria is pending. No A or B result has been computed.
Scope: A (cascade, selective prediction) and B (rank generators, not items). C (judge as drift signal, #163) is out of scope.
Everything is local: models run on the GTX 1660 Ti, no paid API, no Gemini on data, no push.

## Inputs (verified 2026-10-28)
- Judges (unchanged from job 02): MiniCheck-Flan-T5-Large (`minicheck`), Qwen3-4B-Instruct-2507 (4-bit), Phi-4-mini-instruct (4-bit); prompt, 4096-token cap and Yes/No logit as in `judge.py`. `p_yes` = P(faithful). `pred = p_yes >= 0.5`.
- Label convention: 1 = faithful/supported, 0 = not (inverse of RAIM).
- Sonnet has NO predictions here. "Sonnet" below means the RAIM-published Cohen's kappa (RAGTruth 0.601, WiCE 0.548; `analyze.py` RAIM table), a benchmark constant, not a rerun. Caveat inherited from the job-03 report: different subsample, single-token prompt, 4-bit, so ratios are indicative.
- A data: `preds/{minicheck,Qwen_Qwen3-4B-Instruct-2507,microsoft_Phi-4-mini-instruct}__{ragtruth,wice}.jsonl` (samples split only; calib split excluded). All three models have predictions on the identical items: RAGTruth n=300 (150/150), WiCE n=172 (86/86). Verified: line counts match across models. A is fully computable from existing files, CPU only.
- B data: `data/samples/ragtruthgen.jsonl`, built by `build_gen.py` from `wandb/RAGTruth-processed` (download only, already in the local HF cache), QA task, test split, ALL 900 items, 150 per generator, no sampling.
  Why not the existing 300 RAGTruth items: they were built as class-balanced pairs (first clean + first hallucinated answer per query), so each generator's hallucination rate in them is an artefact of the selection (gpt-4: 148 of 300, llama-2-70b: 9 of 300). They cannot rank generators.
  Human reference (hallucination rate = share of items with any hallucination span, test QA): llama-2-7b 52/150 = 34.7%, llama-2-13b 36/150 = 24.0%, llama-2-70b 35/150 = 23.3%, mistral-7B 31/150 = 20.7%, gpt-3.5 5/150 = 3.3%, gpt-4 1/150 = 0.7%. (Train QA, 839 per generator, for reference only: 7b 54.6, 13b 43.3, mistral 41.4, 70b 34.0, gpt-3.5 8.3, gpt-4 4.9: same two tiers, middle order differs.) Written from the dataset labels only, before any judge output on these items exists.
  Sample-size rationale: the test split is the official held-out split, needs no sampling (no selection bias, no seed), and fits the budget. Known limit: the middle four generators differ by 1 to 6 items out of 150, so only the tiers (llama-7b high; the other three llama/mistral mid; gpt-3.5 and gpt-4 low) are resolvable. This limit is part of the design, see B kill rule.
- Compute estimate for the 900 B items: job-02 p50 on RAGTruth was 0.29 s (MiniCheck), 8.5 s (Qwen3-4B), 7.1 s (Phi-4-mini) per item, so about 0.1 h + 2.1 h + 1.8 h = about 4 h, up to 5 h with the longer tail (p95 ~11 s). Run sequentially on the free GPU (3-4 GB VRAM). Prediction only, resumable via `judge.py run --sets ragtruthgen --splits samples`.

## A. Cascade by selective prediction
Hypothesis A: MiniCheck-FT5 can settle a useful share of items alone, at Cohen's kappa >= 0.9x the Sonnet benchmark on those items, and the rest can go to a causal small LM.
Method: confidence of MiniCheck c = |p_yes - 0.5|. For threshold tau, ACCEPTED = {c >= tau} (MiniCheck's own prediction is kept), REJECTED = {c < tau} (forwarded to Qwen3-4B, or Phi-4-mini, as two separate cascades). Fixed cut 0.5, no calib-tuned threshold in the primary analysis.
Metrics (standard):
- Coverage = |accepted| / n. Selective risk = errors among accepted / |accepted| (error = pred != label). Risk-coverage curve = selective risk vs coverage over all tau; AURC = area under it (Geifman & El-Yaniv 2017, "Selective Classification for Deep Neural Networks", NeurIPS). Lower is better; the random-confidence baseline is the overall error rate.
- Cohen's kappa (Cohen 1960): kappa = (po - pe) / (1 - pe), po = observed agreement, pe = chance agreement from the marginals.
- Primary quantity (A1): selective kappa = kappa of MiniCheck on the accepted set. Target = 0.9 x Sonnet kappa = 0.541 (RAGTruth), 0.493 (WiCE). A1 = the largest coverage with selective kappa >= target and |accepted| >= 30 (floor so that tiny subsets cannot fake a result). Caveat: kappa depends on class prevalence and the accepted set is not balanced; report accepted-set prevalence next to every kappa.
- Secondary (A2, success side): cascade kappa on all n items when rejected items take the LM's prediction, at the coverage found in A1, per LM. Success needs A2 >= target. Also report LM calls saved = coverage, and cost = (1 - coverage) x LM p50 latency.
- 95% CI: percentile bootstrap over items, 1000 resamples, seed 7 (same as job 03), for A1 coverage, selective kappa, AURC.
Decision rules (pending owner OK):
- KILL A if A1 coverage < 20% on both RAGTruth and WiCE (point estimate).
- SCALE A if A1 coverage >= 20% on both AND A2 >= target for at least one LM on both sets. The tau sweep is in-sample (optimistic), so a scale call also needs the bootstrap lower bound of A1 coverage > 0 and is reported as "to be confirmed on fresh items".
- Otherwise ITERATE.
Exploratory (not decision-bearing): threshold 0.5 replaced by the calib-tuned cut; confidence from the LMs instead of MiniCheck; a stacked (MiniCheck x LM) confidence; the other three sets (XSum, CNN, ExpertQA); AURC by set.

## B. Rank systems, not items
Hypothesis B: a small judge, applied to a generator's outputs, orders the six RAGTruth generators the way the human labels do.
Method: per generator g and judge j, judged unfaithful rate U(j,g) = share of the generator's 150 items with pred = 0 at the fixed 0.5 cut. Human rate H(g) as above. Ranking quality = Kendall's tau-b between H and U(j,.) over the 6 generators (15 pairs). tau = (concordant - discordant) / (n(n-1)/2); tau-b corrects for ties (Kendall 1938, 1945), needed because human or judged rates can tie.
CI: stratified bootstrap over items within each generator (1000 resamples, seed 7), recomputing BOTH H and U on every resample, then tau-b; percentile 95% interval; resamples where tau-b is undefined (all ties) are dropped and counted.
Decision rules (pending owner OK): KILL B if tau-b < 0.5 or the 95% CI includes 0, for every judge. SCALE B if tau-b >= 0.5 with CI excluding 0 for a judge, and the judge ranks the two tiers (7b, then the 13b/70b/mistral group, then gpt-3.5/gpt-4) correctly. Else ITERATE.
Power caveat stated up front: with n=6 the tau values are multiples of 2/15 and the CI is wide; a pass means the coarse tiers are recovered, not the fine order inside the middle group.
Secondary (exploratory): item-level Cohen's kappa and bacc of each judge on the 900 (natural, imbalanced mix; kappa not comparable to the balanced job-03 numbers); mean(1 - p_yes) as an alternative generator score; Spearman rho as a cross-check; adding the train split (839 per generator) as a second ranking if the test result is borderline. No other metrics are defined.

## Procedure and freeze
1. This file critiqued (agy, anonymized) and corrected; owner OK on A/B kill criteria; then freeze (git tag `prereg-AB-frozen`).
2. B prediction for 900 items may be generated BEFORE freeze, as input preparation only: the run writes `preds/*__ragtruthgen.jsonl` and no metric is computed or looked at.
3. After freeze: `analyze_AB.py` (not yet written) computes A and B once; results written to `report/AB.md`; any deviation is listed there as a deviation.
4. Multiple comparisons: A has 2 sets x 2 LMs, B has 3 judges; the kill rules are conjunctions over sets or judges, so no correction is applied to kill decisions; scale decisions use the bootstrap bounds above. Anything outside these rules is exploratory.
