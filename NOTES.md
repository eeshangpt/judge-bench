# judge-bench NOTES (job 01 setup)

## RAIM (2609.39229) construction, read from refs/2609.39229.txt
- Gold in RAIM: 1 = hallucinated/unsupported, 0 = supported. OURS: label 1 = faithful/supported (inverted).
- Classes exactly 50/50. Sonnet = claude-sonnet-4-6, temp 0, 8192-token context cap.
- RAGTruth: wandb/RAGTruth-processed, train split, task_type=QA; group by query; per query take first response with no
  hallucinated span + first with >=1, same retrieved context. 681 pairs -> 1362 items.
- LLM-AggreFact (WiCE, XSum, CNN, ExpertQA): test split filtered to sub-dataset, classes pooled across docs, majority cut to
  minority count (seed 0). n = 222 / 546 / 114 / 1462.
- Prompt for the claim-support frame (LLM-AggreFact sets) is published (App. K); RAGTruth uses the def_on QA rubric
  (question, retrieved passages, candidate answer). Both ask for a reasoning sentence + VERDICT line; we use a single-token
  Yes/No logprob variant instead (deviation).
- Sonnet kappa (Table 1, panel kappa minus paired delta; read from the Sonnet kappa column): MedHallu 0.761, WiCE 0.548,
  RAGTruth 0.601, XSum 0.418, CNN 0.469, ExpertQA 0.165. Sonnet bacc: MedHallu 88.1, WiCE 77.4, RAGTruth 80.0, XSum 70.9,
  CNN 73.5, ExpertQA 58.2 (verified: stacked kappa - Sonnet kappa equals the paired delta column in each row).
- MiniCheck-Flan-T5-L in RAIM: 66.6 bacc grounded-6, 68.8 on LLM-AggreFact-4.

## Progress log
- env: venv created (.venv, py3.12); driver 595.91 / CUDA 13.2, GTX 1660 Ti 6GB -> torch cu126 wheels.
- env done: torch 2.14.1+cu126, cuda available, GTX 1660 Ti seen. Deps via install_rest.sh (install_torch.sh for torch; network slow, retries).
  MiniCheck loaded straight from HF (lytang/MiniCheck-Flan-T5-Large), logic mirrored from refs/minicheck_inference.py (400-word chunks, max over chunks, logits ids [3,209]).
- data: wandb/RAGTruth-processed gave 681 QA pairs (= RAIM). lytang/LLM-AggreFact is GATED (not read); used public mirror
  NinaCalvi/llm_aggrefact_pre_aug9 (test split). Pool sizes pos/neg: wice 111/247, xsum 285/273, cnn 501/57, expertqa 2971/731
  -> balanced n = 222/546/114/1462, identical to RAIM's. build_data.py writes data/samples + data/calib (seed 7).
  Sizes: ragtruth 300+50, xsum 300+50, expertqa 300+50, wice 172+50, cnn 64+50 (limited by minority class; calib taken first, 25/25).
- judge.py written (minicheck + 4-bit causal, Yes/No logit prompt from RAIM, 4096-token prompt cap vs RAIM 8192).
- prefetch.py downloads weights in background (HF download slow).

## 2026-10-05 day (lead, not job 01)
- 01-setup stopped at 03:11: its last claude turn said "smoke test is running" and ended with no JOB line (no timeout, 0 permission denials, exit success). Setup was otherwise complete: env, data, judge.py, run_local.sh, MiniCheck smoke.
- Smoke Qwen3-4B (ragtruth,wice x5): OK, p_yes sane, 8-54 s/item (median ~13 s). Est ~4.5 h per causal model over 1386 items; 4 causal models = ~18 h, over the 6 h cap.
- Cut: run_local.sh MODELS = minicheck, Qwen3-4B, Phi-4-mini. Qwen2.5-3B and SmolLM3 deferred (add back if time). 01-setup.done marked by hand.
