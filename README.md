# judge-bench

This is the experiment behind the article "A small local judge passed its pilot. Fresh data killed it." It tests whether a small judge model running on a laptop GPU can rank text generators by how often they hallucinate, using human labels as the reference.

## Plans, tags and freeze times

I wrote each plan first, had it reviewed, froze it with a git tag, and only then ran the analysis.

| Plan | Tag | Frozen |
|---|---|---|
| [prereg-AB.md](prereg-AB.md) | `prereg-AB-frozen` | 2026-10-28 02:20 +05:30 |
| [prereg-B2.md](prereg-B2.md) | `prereg-B2-frozen` | 2026-10-28 22:38 +05:30 |

The plans are kept exactly as frozen, including internal references that I did not edit out. #146 and similar refer to my private tracker; "the owner" in the plans is me. Code was written with AI coding assistants; a second model reviewed each plan before it was frozen.

## Results

- `report/AB_B.md`: part B, the pilot on RAGTruth, where the judge ranked generators well enough to go on.
- `report/B2.md`: the FaithBench run on fresh data, where the same judge failed its pre-registered rule.
- `report/AB_A.md`: part A, the cascade test (MiniCheck first, the language model on the rest).
- `figures/`: Figure 1 (`fig1.png`, `fig1.svg`), drawn by `figures/fig1.py` from the tables in `report/B2.md` and `report/AB_B.md`.
- `report/report.md` and `report/results.csv`: an earlier analysis from 2026-10-06 (`analyze.py`), before the two plans above.

## Reproducing

Hardware: one NVIDIA GTX 1660 Ti with 6 GB of memory, Python 3.12, a CUDA build of PyTorch (I used the cu126 wheels). Causal models run in 4-bit.

Models, all from Hugging Face:

- `Qwen/Qwen3-4B-Instruct-2507` (primary judge)
- `lytang/MiniCheck-Flan-T5-Large` (exploratory)
- `microsoft/Phi-4-mini-instruct` (exploratory)

There is no requirements file and I did not pin versions. The packages I installed were torch, transformers, accelerate, bitsandbytes, datasets, scikit-learn, numpy, pandas, sentencepiece and protobuf, plus matplotlib for the figure.

Steps, from the repo root with a virtual environment in `.venv`:

```
.venv/bin/python build_data.py       # RAGTruth and LLM-AggreFact samples -> data/
./run_local.sh                       # part A inputs: all models, five sets, with calibration splits
.venv/bin/python build_gen.py        # RAGTruth QA test items for part B
./run_gen.sh                         # part B predictions
.venv/bin/python -I analyze_AB.py A > report/AB_A.md
.venv/bin/python -I analyze_AB.py B > report/AB_B.md
.venv/bin/python build_b2.py         # FaithBench -> data/samples/faithbench.jsonl
./run_b2.sh                          # B2 predictions
.venv/bin/python -I analyze_B2.py > report/B2.md
.venv/bin/python -I test_AB.py       # synthetic positive/negative controls
.venv/bin/python -I test_B2.py
uv run --with matplotlib python figures/fig1.py   # or any Python with matplotlib
```

`run_local.sh` takes hours on this GPU (about 12 s per item for the causal models). It downloads the model weights through `prefetch.py`. Predictions go to `preds/`, and a rerun skips items already done.

## Data

The datasets are not in this repo, and `data/` is ignored. You need to fetch each one yourself.

- FaithBench: [github.com/vectara/FaithBench](https://github.com/vectara/FaithBench) (file `FaithBench.csv`), paper at [aclanthology.org/2025.naacl-short.38](https://aclanthology.org/2025.naacl-short.38/). License CC BY-NC-SA, so it is not redistributed here. `build_b2.py` expects `data/faithbench/FaithBench.csv`.
- RAGTruth: paper at [arxiv.org/abs/2401.00396](https://arxiv.org/abs/2401.00396). I loaded the processed copy `wandb/RAGTruth-processed` from Hugging Face. Check the dataset card for its license.
- WiCE, and the XSum, CNN and ExpertQA subsets: taken from the LLM-AggreFact collection, loaded as `NinaCalvi/llm_aggrefact_pre_aug9` from Hugging Face. WiCE itself is at [github.com/ryokamoto/wice](https://github.com/ryokamoto/wice). Check each dataset card for its license.

## License

The code is MIT, see `LICENSE`. The datasets keep their own licenses.
