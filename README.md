# Benchmark Saturation Dynamics

This project tests whether a benchmark's **item-difficulty distribution** (shape and variance) decides
when it stops discriminating between models, and whether **evaluation-seed noise** dominates rankings
at saturation. It uses three sources of evidence:

- an IRT model of saturation;
- real per-item results from 220 Open LLM Leaderboard v1 models on 62 tasks;
- a new seed experiment with 15 real open LLMs (10 option-order seeds × 1,500 items) run on local GPUs.

Full details are in **[REPORT.md](REPORT.md)**.

## Key findings
- **Shape does not set the saturation point (H1 not supported).** Matched for SD, uniform, normal,
  Laplace, t₃ and log-normal difficulty distributions all lose discrimination (ρ < 0.5) at nearly the
  same cohort mean accuracy: 95.2–96.6% with 500 items. Real MMLU and HellaSwag items resampled into
  these shapes behave the same way. The saturation accuracy is set mainly by **the number of items**
  (≈80% at N=100, ≈99% at N=2000) and **how close the competing models are**.
- **Variance predicts lifetime only conditionally (H3 only partially supported).** In simulation,
  lifetime correlates with Var(b) at r = 0.92 when the item count is fixed (N≥500). The correlation
  falls to r = 0.20 across item counts and turns negative at N=100. Across the 62 real tasks the
  correlation is r = −0.06 (95% CI −0.38 to 0.33). The r > 0.8 claim is not supported for real
  benchmarks.
- **Seed noise dominates only for near-tied cohorts on small item sets (H2 is conditional).** Seed
  noise from option order measured on real LLMs is about 0.40× binomial item-sampling noise. On full
  benchmarks and the real leaderboard top-20, seeds explain at most about 18% of score variance. The
  seed share rises steeply with accuracy (Spearman 0.96), but reaches 0.5 only when cohorts are close
  in capability and accuracy is ≥ 99% on 150-item subsets.
- **Real leaderboard history:** HellaSwag (10k items) stayed reliable at about 89% frontier accuracy.
  Smaller benchmarks such as Winogrande and ARC were less reliable at lower accuracy.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv sync          # dependencies in pyproject.toml
python src/download_v1_details.py      # per-item leaderboard data (~45 min, HF rate limits)
python src/build_matrices.py           # model x item matrices
python src/exp1_irt_saturation.py      # Exp 1 (CPU, ~10-30 min)
python src/exp2_real_benchmarks.py     # Exp 2 (GPU optional)
src/run_seed_eval.sh <gpu> <hf-model> ...   # Exp 3 data (GPU)
python src/exp3_seed_analysis.py && python src/h2_crosscheck.py
python src/exp4_leaderboard_history.py
python src/make_figures.py
```

## Layout
```
src/            experiment & analysis code (see docstrings)
datasets/       downloaded leaderboard tables, per-item pickles, ARC/MMLU samples (git-ignored data)
results/        exp1/ exp2/ exp3/ exp4/ seed_eval/ (raw model outputs) matrices/
figures/        all report figures
planning.md     pre-registered plan; REPORT.md  full report
```
