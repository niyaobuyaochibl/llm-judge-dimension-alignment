# Dimension Alignment in LLM-as-a-Judge Recommendation Evaluation

Code, experiment outputs, and analysis for the paper:

> Yunan Zhang, Jingjing Fan, and Yanxiao Liu.
> **What Do LLM Judges Evaluate in Recommendation Assessment? Disentangling Relevance, Local Diversity, and System-Level Diversity.**
> Accepted for publication in *IEEE Transactions on Knowledge and Data Engineering*, 2026.

When an LLM judge declares one recommendation list "better" than another, which quality
dimension drives that verdict? This repository contains everything needed to reproduce the
answer reported in the paper: 19,240 LLM calls across four datasets, 16 model pairs, four
prompt designs, four LLM judges (7B–671B), four frontier closed-source judges, four
diagnostic prompt controls, and a small human sanity check. It also contains the code and
seed-42 checkpoints of the recommenders whose lists were judged (see
[Recommender models](#recommender-models-and-checkpoints)).

## Reproducing the paper without a GPU

Every number, table, and figure in the paper is derived from the stored judge verdicts in
`results/`. Regenerating them needs no GPU, no API key, and no dataset download:

```bash
pip install -r requirements.txt
cd code
python minor_revision_analysis.py     # writes the 14 CSVs in revision_results/
python analyze_results.py             # main-text alignment tables and Figure 1
python supplementary_analysis.py      # multi-metric and per-dataset breakdowns
python mixed_effects_analysis.py      # GEE logistic regression
python fdr_correction.py              # Benjamini-Hochberg correction over the 17 tests
```

`revision_results/local_metric_denominators.csv` is the 32-cell table behind Table IX: for
every prompt-by-metric cell it lists the consistent verdicts, the eligible denominator, the
aligned count, metric ties, and unavailable cases.

## Layout

```
code/                  26 experiment and analysis scripts, plus two shared modules
results/               judge verdicts, 233 files
  *.json                 Qwen2.5-7B, 4 datasets x 4 pairs x 4 prompts
  14b/ mistral/ deepseek/    cross-scale and cross-family validation
  openrouter_validation_*/   GPT-4o, GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro
  revision_prompts_*/        few-shot, structured, and global-statistics controls
  revision_pointwise*/       pointwise local-diversity scoring control
  sensitivity_n100/          N = 100 users per pair sensitivity check
  *_smoke*/                  small pilot runs, kept for transparency
revision_results/      the 14 analysis CSVs cited in the paper
human_study/           330 annotations from 5 annotators, plus the protocol
data/topk_results/     top-10 lists produced by every recommender (item IDs)
recsys/                recommender training/evaluation code, 22 configs, checkpoint checksums
```

## Full reproduction from scratch

This path re-runs the judges and needs the source datasets, a GPU, and API keys.

**Step 0.** Obtain the source datasets (MIND, Yelp, and two Amazon categories) and build the
processed files. See [`data/README.md`](data/README.md); they are not redistributed here.

**Step 1.** Point the code at your copies by editing the five path constants at the top of
`code/experiment_config.py`:

```python
PROJECT_ROOT    = Path("...")   # this repository
DIVERSITY_ROOT  = Path("...")   # processed datasets
LLM_JUDGE_ROOT  = Path("...")   # shared utilities (also vendored into code/)
MODEL_PATH_7B   = "..."         # Qwen2.5-7B-Instruct
MODEL_PATH_14B  = "..."         # Qwen2.5-14B-Instruct
```

Several `run_*.py` scripts also carry an absolute `sys.path` insert pointing at the original
machine. `data_loader.py` and `judge_pipeline.py` are vendored into `code/`, so those inserts
resolve harmlessly on a fresh clone; remove them if you prefer.

**Step 2.** Run the experiments.

```bash
cd code
python run_dimension_experiments.py        # Qwen2.5-7B, the main 4x4x4 grid
python run_14b_validation.py               # Qwen2.5-14B, 4-bit
python run_mistral_validation.py           # Mistral-7B-Instruct-v0.3
DEEPSEEK_API_KEY=...   python run_deepseek_validation.py
OPENROUTER_API_KEY=... python run_openrouter_validation.py
python run_revision_prompt_experiments.py  # few-shot / structured / global-stat controls
python run_revision_pointwise_scoring.py   # pointwise local-diversity scoring
python run_sensitivity_n100.py             # N = 100 sensitivity check
```

No API key is stored in this repository; the two API scripts read them from the environment.

## Recommender models and checkpoints

The judges compared top-10 lists from six recommenders per dataset: GBAF, Fixed, Concat-MLP,
GradNorm, PCGrad, and LightGCN. Their training and evaluation code is in
[`recsys/`](recsys/), and the 22 seed-42 checkpoints that produced every list in
`data/topk_results/` are attached to the
[GitHub release](https://github.com/niyaobuyaochibl/llm-judge-dimension-alignment/releases)
as four per-dataset archives (418 MB in total), with SHA-256
checksums in `recsys/CHECKPOINT_SHA256SUMS.txt`. `recsys/README.md` maps each checkpoint to
its config and shows how to regenerate the lists from a checkpoint or retrain from scratch.

## Experiment scale

| Component | Calls |
|---|---|
| Phase 1, balanced prompt | 1,600 |
| Phase 2, dimension prompts | 4,800 |
| Qwen2.5-14B validation | 1,800 |
| Mistral-7B validation | 1,800 |
| DeepSeek-V3 API validation | 1,800 |
| Four diagnostic prompt controls | 2,400 |
| N = 100 sensitivity check | 3,600 |
| Frontier closed-source API validation | 1,440 |
| **Total** | **19,240** |

## Human sanity check

`human_study/` holds the 330 annotations behind Table XI. Five annotators each judged 60
local-only diversity cases and 6 global-statistics cases. Annotators were informed adult
volunteers; no names, demographics, or other identifying information were collected, and every
task displayed only anonymized recommendation lists.

## Citation

```bibtex
@article{zhang2026dimension,
  title   = {What Do {LLM} Judges Evaluate in Recommendation Assessment?
             Disentangling Relevance, Local Diversity, and System-Level Diversity},
  author  = {Zhang, Yunan and Fan, Jingjing and Liu, Yanxiao},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year    = {2026},
  note    = {Accepted; volume, issue, and DOI to follow}
}
```

## License

Code and analysis outputs are released under the MIT License (see `LICENSE`). The source
recommendation datasets keep their own licenses and are not redistributed here.
