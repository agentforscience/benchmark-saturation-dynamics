"""Exp 2: saturation dynamics on real LLM response data (Open LLM Leaderboard v1, 62 tasks).

Models are split into a *calibration* half (used only to estimate item difficulties) and an
*evaluation* half (used to measure discrimination), alternating along the external-ability ranking,
so item selection/difficulty estimation never sees the responses used to measure discrimination.

2a  Per-task item difficulty distribution (Rasch with fixed guessing floor, calibration half).
2b  Empirical discrimination along the ability axis: sliding cohorts of K=20 evaluation models
    (sorted by an external ability composite that excludes the task's own benchmark); per cohort,
    split-half reliability (Spearman-Brown) on fixed-size item subsamples (N=90).
    Empirical lifetime = fraction of cohorts whose reliability >= 0.5.
2c  Model-based lifetime: Exp-1 reliability curve evaluated with the task's *real* fitted
    difficulties (uncensored by the observed ability range).
2d  Correlation of lifetime with difficulty variance across the 62 tasks (H3).
2e  Real-item resampled benchmarks with prescribed difficulty shapes (H1) from the MMLU and
    HellaSwag item pools; discrimination vs cohort mean accuracy.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from irt import fit_irt, guess_floor  # noqa: E402
from exp1_irt_saturation import curves, cohort, lifetime  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT = os.path.join(ROOT, "results", "matrices")
RES = os.path.join(ROOT, "results", "exp2")
os.makedirs(RES, exist_ok=True)
SEED = 42
K, STEP, N_SUB, N_SPLITS = 20, 2, 90, 20
DEV = "cuda" if __import__("torch").cuda.is_available() else "cpu"
BENCHES = ["arc", "hellaswag", "truthfulqa", "winogrande", "gsm8k", "mmlu"]


def bench_of(task):
    return "mmlu" if task.startswith("mmlu_") else task


def load():
    tasks = sorted(f[:-8] for f in os.listdir(MAT) if f.endswith(".parquet"))
    mats = {t: pd.read_parquet(os.path.join(MAT, f"{t}.parquet")) for t in tasks}
    models = sorted(set.intersection(*[set(m.index) for m in mats.values()]))
    mats = {t: m.loc[models] for t, m in mats.items()}
    # benchmark-level accuracies (MMLU = item-weighted over subjects)
    acc = {}
    for bch in BENCHES:
        ts = [t for t in tasks if bench_of(t) == bch]
        tot = sum(mats[t].sum(1) for t in ts)
        n = sum(mats[t].notna().sum(1) for t in ts)
        acc[bch] = tot / n
    acc = pd.DataFrame(acc)
    z = (acc - acc.mean()) / acc.std()
    return tasks, mats, acc, z


def split_half_rel(S, rng, n_splits=N_SPLITS):
    """S: (models, items) 0/1 for a cohort. Mean Spearman-Brown corrected split-half reliability."""
    n = S.shape[1]
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        h1, h2 = S[:, perm[: n // 2]].mean(1), S[:, perm[n // 2:]].mean(1)
        if h1.std() == 0 or h2.std() == 0:
            rs.append(0.0)
            continue
        r = np.corrcoef(h1, h2)[0, 1]
        rs.append(2 * r / (1 + r) if r > -1 else -1)
    return float(np.mean(rs))


def binom_rel(S):
    """Reliability vs. binomial (item/response sampling) noise: 1 - mean p(1-p)/N / Var(p)."""
    p = S.mean(1)
    v_obs = p.var(ddof=1)
    noise = np.mean(p * (1 - p) / S.shape[1])
    return float(max(0.0, 1 - noise / v_obs)) if v_obs > 0 else 0.0


def windows(order):
    return [order[i:i + K] for i in range(0, len(order) - K + 1, STEP)]


def main():
    rng = np.random.default_rng(SEED)
    tasks, mats, acc, z = load()
    models = list(acc.index)
    print(f"{len(models)} models x {len(tasks)} tasks")

    task_rows, cohort_rows = [], []
    for task in tasks:
        bch = bench_of(task)
        ext = z.drop(columns=[bch]).mean(1)                    # external ability (leave-benchmark-out)
        order = ext.sort_values().index.tolist()
        calib, evalm = order[0::2], order[1::2]
        X = mats[task]
        c = 0.0  # plain Rasch: below-chance (adversarial-distractor) items make a fixed guessing floor misfit
        fit = fit_irt(X.loc[calib].values, c=c, rasch=True, iters=300, device=DEV)
        b = fit["b"]
        th = fit["theta"]
        q = np.quantile(b, [0.05, 0.25, 0.75, 0.95])
        th_cal_sorted = th  # calib list is in external-ability order
        p_items = X.loc[calib].mean(0).clip(0.01, 0.99).values
        logit_p = np.log(p_items / (1 - p_items))
        # ---- 2b empirical cohort discrimination (evaluation half)
        E = X.loc[evalm].values
        rels, accs, brels = [], [], []
        for w_i, w in enumerate(windows(list(range(len(evalm))))):
            r_sh, r_b, a_m = [], [], []
            for _ in range(5):  # item subsamples of fixed size
                cols = rng.choice(E.shape[1], min(N_SUB, E.shape[1]), replace=False)
                S = np.nan_to_num(E[np.ix_(w, cols)], nan=0.0)
                r_sh.append(split_half_rel(S, rng, 8))
                r_b.append(binom_rel(S))
                a_m.append(S.mean())
            rels.append(np.mean(r_sh)); brels.append(np.mean(r_b)); accs.append(np.mean(a_m))
            cohort_rows.append(dict(task=task, window=w_i, ext_center=float(ext.loc[[evalm[k] for k in w]].mean()),
                                    acc=accs[-1], rel_sh=rels[-1], rel_binom=brels[-1]))
        rels, brels = np.array(rels), np.array(brels)
        # ---- 2c model-based lifetime with real difficulties (N_SUB items, cohort spread from data)
        th_sd = th.std()
        # cohort spread: median within-window SD of calibrated theta (same K-window construction)
        cw = windows(list(range(len(calib))))
        th_sorted = th  # calib already in ext order
        sig_c = float(np.median([th_sorted[w].std(ddof=1) for w in cw]))
        t_grid = np.linspace(b.min() - 6, b.max() + 6, 300)
        lifes = []
        for _ in range(10):
            bs = rng.choice(b, min(N_SUB, len(b)), replace=False)
            df = curves(bs, np.ones_like(bs), c, cohort(K, sig_c), t_grid)
            lifes.append(lifetime(df, "rho", 0.5) / th_sd)       # in units of population theta-SD
        task_rows.append(dict(
            task=task, bench=bch, n_items=X.shape[1], guess=c, mean_acc=float(np.nanmean(X.values)),
            mean_acc_top20=float(np.nanmean(X.loc[order[-20:]].values)),
            var_b=float(b.var()), sd_b=float(b.std()), iqr_b=float(q[2] - q[1]), range90_b=float(q[3] - q[0]),
            skew_b=float(stats.skew(b)), kurt_b=float(stats.kurtosis(b)), mean_b=float(b.mean()),
            var_b_std=float(b.var() / th_sd ** 2),               # difficulty variance in population-SD units
            var_logit_p=float(logit_p.var()), theta_sd=float(th_sd), sigma_c=sig_c,
            life_emp=float((rels >= 0.5).mean()), life_emp_binom=float((brels >= 0.5).mean()),
            life_model=float(np.mean(lifes)), max_rel=float(rels.max()),
            rel_top=float(rels[-1])))
        print(f"{task:45s} acc={task_rows[-1]['mean_acc']:.2f} var_b={b.var():.2f} "
              f"life_emp={task_rows[-1]['life_emp']:.2f} life_model={np.mean(lifes):.2f}", flush=True)
    T = pd.DataFrame(task_rows)
    C = pd.DataFrame(cohort_rows)
    T.to_csv(os.path.join(RES, "task_stats.csv"), index=False)
    C.to_csv(os.path.join(RES, "cohort_curves.csv"), index=False)

    # ---- 2d correlations (H3) with bootstrap CIs
    out = {}
    for y in ["life_emp", "life_emp_binom", "life_model"]:
        for x in ["var_b", "var_b_std", "sd_b", "var_logit_p", "iqr_b", "range90_b", "kurt_b"]:
            r, p = stats.pearsonr(T[x], T[y])
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(T), len(T))
                boots.append(np.corrcoef(T[x].values[idx], T[y].values[idx])[0, 1])
            out[f"{x}~{y}"] = dict(pearson=float(r), p=float(p), ci95=[float(np.nanpercentile(boots, 2.5)),
                                   float(np.nanpercentile(boots, 97.5))],
                                   spearman=float(stats.spearmanr(T[x], T[y]).statistic))
        # partial correlation controlling for mean accuracy (task location / censoring)
        import statsmodels.api as sm
        rx = sm.OLS(T["var_b"], sm.add_constant(T[["mean_acc"]])).fit().resid
        ry = sm.OLS(T[y], sm.add_constant(T[["mean_acc"]])).fit().resid
        out[f"partial(var_b~{y}|mean_acc)"] = float(np.corrcoef(rx, ry)[0, 1])
    json.dump(out, open(os.path.join(RES, "lifetime_correlations.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))

    # ---- 2e real-item resampled benchmarks (H1)
    resample_rows = []
    for pool_name, pool_tasks in [("mmlu", [t for t in tasks if t.startswith("mmlu_")]), ("hellaswag", ["hellaswag"])]:
        X = pd.concat([mats[t].add_prefix(t + ":") for t in pool_tasks], axis=1)
        ext = z.drop(columns=[pool_name]).mean(1)
        order = ext.sort_values().index.tolist()
        calib, evalm = order[0::2], order[1::2]
        fit = fit_irt(X.loc[calib].values, c=0.0, rasch=True, iters=300, device=DEV)
        b = fit["b"]
        kde = stats.gaussian_kde(b)
        f_pool = kde(b)
        E = X.loc[evalm].values
        np.save(os.path.join(RES, f"pool_b_{pool_name}.npy"), b)
        sd = 1.0
        for fam in ["uniform", "normal", "laplace", "t3", "lognormal_hard"]:
            for m in np.linspace(np.quantile(b, 0.05), np.quantile(b, 0.6), 12):
                for rep in range(6):
                    zz = (b - m) / sd
                    if fam == "uniform":
                        f_t = (np.abs(zz) <= np.sqrt(3)).astype(float)
                    elif fam == "normal":
                        f_t = stats.norm.pdf(zz)
                    elif fam == "laplace":
                        f_t = stats.laplace.pdf(zz, scale=1 / np.sqrt(2))
                    elif fam == "t3":
                        f_t = stats.t.pdf(zz * np.sqrt(3), 3)
                    else:  # right-skewed lognormal standardised
                        x = zz * np.sqrt((np.e - 1) * np.e) + np.exp(0.5)
                        f_t = np.where(x > 0, stats.lognorm.pdf(np.maximum(x, 1e-9), 1), 0.0)
                    w = f_t / f_pool
                    if (w > 0).sum() < 250:
                        continue
                    sel = rng.choice(len(b), 200, replace=False, p=w / w.sum())
                    bs = b[sel]
                    for w_i, win in enumerate(windows(list(range(len(evalm))))):
                        if w_i % 3 and w_i != len(windows(list(range(len(evalm))))) - 1:
                            continue  # thin the cohorts for speed (keep top cohort)
                        S = np.nan_to_num(E[np.ix_(win, sel)], nan=0.0)
                        resample_rows.append(dict(pool=pool_name, family=fam, loc=float(m), rep=rep, window=w_i,
                                                  sd_b=float(bs.std()), kurt_b=float(stats.kurtosis(bs)),
                                                  skew_b=float(stats.skew(bs)), acc=float(S.mean()),
                                                  rel_sh=split_half_rel(S, rng, 10), rel_binom=binom_rel(S),
                                                  tau_ext=float(stats.kendalltau(S.mean(1), ext.loc[[evalm[k] for k in win]].values).statistic)))
        print("resampling done for", pool_name, flush=True)
    R = pd.DataFrame(resample_rows)
    R.to_csv(os.path.join(RES, "resampled_benchmarks.csv"), index=False)


if __name__ == "__main__":
    main()
