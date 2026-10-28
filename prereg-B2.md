# Pre-registration B2, issue #146: fresh-data confirmation of B on FaithBench

Status: v3 FROZEN on tag `prereg-B2-frozen`. Audit outcome: agy round 3 CHANGES, owner decision freeze 2026-10-28; open findings listed (strict tier rule and point-estimate gate, tiers built from this dataset own rates, same-family and filter-generator checks not gating, uncalibrated 0.5 cut, undefined tau-b imputed as 0, pooled item-level metrics, generator-level permutation null). Local only, nothing pushed. Owner approved the plan on 2026-10-28.
Scope: does a small local judge rank summarizers by hallucination rate the way human labels do, on data it was not developed on? B (RAGTruth, 6 generators, `prereg-AB.md`) said SCALE as a pilot; B2 asks the same question on a second dataset.

## Data
- FaithBench (Bao et al., NAACL 2025, Vectara): `FaithBench.csv`, 800 rows = 10 summarizers x 80 summaries; columns source, summary, LLM, worst-label, best-label. Fetched 2026-10-28 by anonymous HTTPS GET from `raw.githubusercontent.com/vectara/FaithBench/main/FaithBench.csv` into `data/faithbench/` (git-ignored). Licence CC BY-NC-SA: the data is never committed to or published in any repo.
- Label meaning (confirmed from the FaithBench README and `scripts/binarize.py`): each summary was annotated by several annotators, each marking spans with labels Benign, Questionable, Unwanted (Unwanted has Intrinsic/Extrinsic sub-types). The sample-level `worst-label` pools all labels from all annotators and takes the most severe (severity Benign 1 < Questionable 2 < Unwanted 3); no annotation = Consistent. Counts: Unwanted 485, Consistent 174, Questionable 77, Benign 64.
- Primary definition: unfaithful = `worst-label == "Unwanted"`. Sensitivity (descriptive): unfaithful = Unwanted or Questionable.
- Human rate H(g) = share of the generator's 80 summaries that are unfaithful. Tiers are FIXED NOW from the human Unwanted rates (verified from the CSV, `build_b2.py` stops if a rate differs from the list by more than rounding):
  - Low (6): gpt-4o 0.4625, GPT-3.5-Turbo 0.475, Llama-3.1-8B 0.55, gemini-1.5-flash 0.5625, claude-3-5-sonnet 0.575, Llama-3.1-70B 0.5875.
  - High (4): Mistral-7B-v0.3 0.70, Phi-3-mini 0.7125, command-r 0.7125, Qwen2.5-7B 0.725.
- Disclosure: the human rates and the tier split were seen before this freeze. No judge output on FaithBench exists (no judge has seen any FaithBench item); this prereg is committed and tagged before inference starts.

## Judges
- PRIMARY: Qwen3-4B-Instruct-2507 (4-bit), exactly the job-02/B harness: claim-support prompt (`CLAIM_PROMPT`, source = document, summary = claim), single-token Yes/No logit, `p_yes` = P(supported), cut 0.5, 4096-token prompt cap. A summary is judged unfaithful iff `p_yes < 0.5`.
- EXPLORATORY: MiniCheck-Flan-T5-Large (same code as before). Phi-4-mini is not run (owner decision).
- No tuning, no threshold change, no prompt change after seeing FaithBench results.

## Metrics (standard names)
- Judged unfaithful rate U(j,g) = share of generator g's 80 items with `p_yes < 0.5`.
- Kendall's tau-b between H and U over the 10 generators (45 pairs): with P concordant and Q discordant pairs, n0 = n(n-1)/2, n1 = sum over tied groups in H of t(t-1)/2, n2 = same for U, tau-b = (P - Q) / sqrt((n0 - n1)(n0 - n2)) (Kendall 1945; tau-a is Kendall 1938).
- Tier order correct iff min U(High) > max U(Low) (strict).
- Design fact found after the owner plan (checked in the CSV): the 800 rows are 80 source documents, each summarized by all 10 LLMs (matched design). PRIMARY CI therefore = cluster bootstrap over the 80 documents (1000 resamples, seed 7; each sampled document brings its 10 summaries), recomputing BOTH H and U each time; percentile 95%. This deviates from the plan's 'stratified within generator', which ignores the pairing; the stratified-within-generator CI is reported as a sensitivity. Undefined tau-b counts as 0 in the primary CI; the CI with them dropped and their count are reported. No decision rule uses a CI.
- Exact two-sided permutation p of the observed tau-b (descriptive; it permutes the 10 generator-level rates, so it tests a marginal exchangeability null and ignores the document pairing): all 10! = 3,628,800 orderings of U against fixed H, p = share with |tau-b| >= |observed|.
- Descriptive only: within-tier tau-b (High, Low separately); tier separation AUC = (number of the 24 High-Low generator pairs with U(High) > U(Low) + 0.5 x number of ties) / 24 (Mann-Whitney); tau-b using the threshold-free score mean(1 - p_yes) per generator (guards against the uncalibrated 0.5 cut); Spearman rho; tau-b against the sensitivity H' (Unwanted or Questionable), tiers unchanged (human rates under H' checked from the CSV before freezing: the same four generators are the top four, 0.775-0.825, and the highest Low generator is 0.70, so the split still holds); item-level Cohen's kappa (Cohen 1960) and balanced accuracy (Brodersen et al. 2010) on the 800 items (kappa = (po - pe)/(1 - pe), bacc = (TPR + TNR)/2).

## Decision rule (primary judge only; same as B)
- SCALE if tau-b >= 0.5 AND tier order correct. SCALE here means: B replicates on a second dataset, worth a third test, not a deployment claim.
- KILL if tier order is not correct OR tau-b < 0.3 (or undefined).
- ITERATE otherwise (0.3 <= tau-b < 0.5 with tiers right): one more round only, re-registered as a new prereg, allowed to change only the prompt, the cut (e.g. a calibrated one), or the judge model; no third round without owner OK.
- MiniCheck is reported with the same rule but gates nothing.

## Exploratory (not decision-bearing)
- MiniCheck verdict; the sensitivity definition H'; Spearman; item-level kappa and bacc; a same-family check with the Qwen2.5-7B generator dropped (9 generators, High tier = 3): tau-b and CI, tier-order recomputed; a filter-generator check with gpt-4o and GPT-3.5-Turbo dropped (8 generators, Low tier = 4): tau-b, tier order; Spearman correlation between U and mean summary length in words per generator (length confound, descriptive). Neither check changes the verdict, which uses the primary 10-generator rule only.

## Known limits (stated before the run)
1. Selection: FaithBench items were chosen where SOTA detectors disagreed (highest entropy among GPT-4o-judge, GPT-3.5-judge, TrueTeacher, True-NLI, HHEM-2.1). The data is adversarial, so overall rates (about 60% Unwanted) and detector behaviour are not typical; item-level numbers do not transfer to natural data. Rates per generator are conditional on being hard for those detectors, and two of the five detectors (GPT-4o, GPT-3.5) are also generators in the list.
2. Power: n = 80 per generator gives a binomial 95% CI of about +-11 points on each rate near 0.6. The Low tier spans 0.46-0.59 and the High tier 0.70-0.73, so the tier gap (0.59 vs 0.70) is about the CI width; only the coarse tier split is testable, not the order inside a tier (differences of 1-6 items).
3. Same-family risk: Qwen2.5-7B is a generator in the High tier and the primary judge is Qwen3-4B (Qwen family). The dropped-generator check is exploratory.
4. Truncation: any item over the context limit is counted and reported (`truncated` flag in the predictions); FaithBench sources are short (max about 950 words), so zero is expected.
5. Small models were developed on RAGTruth-like prompts in B; human rates here span only 0.46-0.73, a much smaller contrast than RAGTruth's 0.7%-35%.
6. Scope: B2 tests generator ranking on an adversarial, detector-filtered set of 80 documents, not natural-distribution ranking; a pass says the ranking ability replicates on hard cases, nothing about natural data. The prompt is the claim-support prompt applied to whole multi-sentence summaries (as in the AggreFact sets), with an uncalibrated 0.5 cut; the mean(1 - p_yes) check above is the threshold-free view.
7. The tier split is built from the human rates of this very dataset (the gap 0.59 to 0.70 is an observed one), so the tier-order check alone is partly circular with H; that is why tau-b over all 10 generators is a second, separate criterion, and why the tier rule is strict. Strict tier rule: one boundary tie or inversion kills (owner-fixed rule); the AUC and within-tier tau-b are given so the reader can see how close a KILL was.
8. Conclusions are conditional on this one sample of 80 source documents; documents are shared across generators, so document-level correlation limits generalisation beyond them.
9. Single run of the analysis; any deviation from this document is listed as a deviation in `report/B2.md`.

## Procedure
1. Critique by agy (anonymized, no tools), corrections applied, owner plan already approved.
2. Commit, tag `prereg-B2-frozen`, then `run_b2.sh` (inference only) on the local GPU, no paid API.
3. `analyze_B2.py` once -> `report/B2.md`; synthetic positive/negative controls (`test_B2.py`) pass before the freeze.
4. agy audit of `report/B2.md` plus this file, same prompt each round, max 3 rounds.
