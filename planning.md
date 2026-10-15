# Planning: Benchmark Saturation Dynamics

## Motivation & Novelty Assessment

### Why This Research Matters
Leaderboards drive model development, funding and deployment decisions, yet benchmarks
"saturate": once frontier models cluster near the ceiling, score differences reflect noise
(evaluation seeds, prompt formatting, item sampling) rather than capability. Knowing *when* a
benchmark stops discriminating — and which design property (item-difficulty distribution) delays
that point — tells benchmark builders how to construct longer-lived benchmarks and tells
practitioners when to stop trusting rank differences.

### Gap in Existing Work
No pre-gathered literature review was available in this workspace (the `literature_review.md`,
`papers/`, `datasets/`, `code/` resources were absent), so this assessment relies on the author's
knowledge of the field. Prior work shows: (i) IRT can compress benchmarks (tinyBenchmarks, Polo et
al. 2024; metabench, Kipnis et al. 2024) and items differ greatly in informativeness;
(ii) evaluation scores have non-trivial seed / prompt variance (Madaan et al. 2024 "Quantifying
Variance in Evaluation Benchmarks"; Miller 2024 "Adding Error Bars to Evals"); (iii) benchmarks
saturate quickly (Kiela et al. 2021 Dynabench; Ott et al. 2022 "Mapping global dynamics of benchmark
creation and saturation"). What is missing is a *quantitative link* between the **shape/variance of
the item difficulty distribution** and the **accuracy level at which discrimination collapses**,
plus a direct comparison of **seed-induced vs capability-induced ranking variance as a function of
saturation level**.

### Our Novel Contribution
1. A psychometric (IRT) analysis of discrimination vs mean accuracy for benchmarks whose difficulty
   distributions differ in shape (uniform, normal, heavy-tailed) and in variance, separating shape
   effects from variance effects.
2. A real-data test using per-item correctness of ~220 Open LLM Leaderboard v1 models on 62 tasks
   (5 benchmarks + 57 MMLU subjects), estimating each task's difficulty variance and its "effective
   lifetime" (ability range over which it reliably ranks contemporaneous models).
3. A real-model seed experiment (local open LLMs, CPU) decomposing ranking variance into
   capability vs evaluation-seed components at different saturation levels.

### Experiment Justification
- **Exp 1 (IRT theory/simulation, calibrated to real data):** the only setting where difficulty
  distribution can be manipulated with everything else fixed; tests H1 (uniform saturates at 85% vs
  heavy-tailed at 95%) and H3 (lifetime vs difficulty variance) under controlled conditions.
  Note: this simulates *item responses from an IRT model*, not LLM behaviour; it is paired with
  real-data experiments.
- **Exp 2 (real leaderboard responses):** tests whether the predictions hold for real benchmarks:
  (a) per-task lifetime vs difficulty variance correlation (H3) across 62 real tasks;
  (b) resampled sub-benchmarks built from real items with uniform vs heavy-tailed difficulty
  distributions, scored with real model responses (H1).
- **Exp 3 (real seed variance):** H2 requires actual seed-to-seed variability of real models.
  We run open LLMs locally with several evaluation seeds (answer-option permutation + sampling)
  on easy/medium/hard tasks and compute the variance share of seeds vs models.
- **Exp 4 (historical leaderboard dynamics):** uses dated v1 leaderboard scores (7k submissions)
  to show real saturation trajectories (top-k spread over calendar time) for the 6 benchmarks.

## Research Question
Does the shape/variance of a benchmark's item difficulty distribution determine the mean accuracy at
which it stops discriminating between models, and does evaluation-seed noise dominate ranking
variance at saturation?

## Hypothesis Decomposition
- H1: For uniform difficulty distributions, discrimination (rank reliability among a cohort of
  contemporaneous models) falls below threshold once cohort mean accuracy > ~85%; heavy-tailed
  distributions keep discriminating up to ~95%.
- H2: At saturation, evaluation-seed variance accounts for more of the variance in rankings than true
  between-model capability differences (seed share > 50%).
- H3: Effective lifetime correlates with variance of item difficulty with r > 0.8.

Independent variables: difficulty distribution shape & variance, cohort ability level (≈ time).
Dependent variables: rank reliability / Kendall τ with true ability, fraction of separable pairs,
seed variance share, lifetime.

## Proposed Methodology
### Definitions
- Item difficulty b_i: IRT (Rasch/2PL-style) difficulty; on real data, estimated by joint maximum
  likelihood Rasch fit per task (logit scale) on all models.
- Cohort: K contemporaneous models with abilities within a window (sim: θ ~ N(μ_t, σ_c);
  real: sliding window over models sorted by an external ability composite).
- Discrimination D: (sim) Kendall τ between observed scores and true θ; also reliability
  ρ = Var(true score)/Var(observed score). (real) split-half reliability of cohort scores,
  Spearman–Brown corrected, using equal-size item subsamples.
- Saturation point: cohort mean accuracy at which D falls below threshold (τ<0.5 primary; ρ<0.5).
- Effective lifetime: length of the ability interval (logits; maps to time under linear progress)
  over which D ≥ threshold.

### Experimental Steps
1. Exp 1: Rasch/2PL simulation; 5 distribution families (point-mass/narrow, uniform, normal,
   Laplace, Student-t3, log-normal-skewed hard tail) × matched-variance and natural-variance
   settings; cohort of K=20 models; N=500 items; 200 replicates per cohort level; sweep μ_t.
   Lifetime vs SD(b) across 300 random benchmark configs → Pearson r.
2. Exp 2: download per-item results; build response matrices; fit Rasch per task; compute
   difficulty variance; sliding-cohort split-half reliability vs cohort mean accuracy; lifetime;
   correlation (with bootstrap CI), controlling for #items. Resampled real-item benchmarks with
   prescribed difficulty distributions.
3. Exp 3: local LLMs (Qwen2.5 0.5B/1.5B/3B, Qwen3, SmolLM2, Phi-3.5 …) on ARC-Easy, ARC-Challenge,
   MMLU subset; S seeds (option permutation, temperature-1 sampling); variance components via
   two-way decomposition; ranking flip rates; repeated for saturated vs non-saturated subsets.
4. Exp 4: leaderboard history: monthly top-k spread & rank stability per benchmark.

### Baselines
- Point-mass ("all items equally hard") and normal distributions as reference shapes.
- Random-item-subset benchmarks (natural difficulty distribution) as baseline for resampling.
- Bootstrap item-sampling noise as baseline noise source for seed analysis.

### Statistical Analysis Plan
Pearson & Spearman correlations with bootstrap 95% CIs (2000 resamples); partial correlation
controlling for number of items/base rate; variance components with bootstrap CIs; α = 0.05;
Holm correction where multiple tests are reported together.

## Expected Outcomes
Supporting H1: crossover accuracy uniform ≈ 85%, heavy-tailed ≈ 95%. Refuting: similar crossover,
or ordering driven only by variance not shape. H2: seed share > 0.5 when cohort mean acc ≥ 90%.
H3: r > 0.8 between lifetime and Var(b).

## Timeline
Setup+data 45 min; Exp1 30 min; Exp2 60 min; Exp3 90 min (CPU-bound, runs in background);
Exp4 20 min; analysis+report 60 min.

## Potential Challenges
- No API keys → real-LLM seed experiment must run locally on CPU (small models only).
- v2 leaderboard details gated → use v1 (public).
- Real tasks measure different abilities; cohort ordering uses an external composite (leave-task-out).
- Rasch is a simplification (no guessing parameter); 4-option MC has 25% chance floor — handled by
  3PL-style guessing in simulation sensitivity check.

## Success Criteria
All three sub-hypotheses tested with quantitative criteria on both controlled and real data, with
honest reporting of where they hold or fail.
