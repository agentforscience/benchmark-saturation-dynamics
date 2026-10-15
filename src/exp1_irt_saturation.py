"""Exp 1: IRT model of benchmark saturation under different item-difficulty distributions.

Item responses follow a 2PL/3PL model  P(correct | theta, b, a, c) = c + (1-c) * sigmoid(a (theta - b)).
A "cohort" of K contemporaneous models has abilities theta_k = t + sigma_c * z_k (z fixed, evenly
spaced normal quantiles) and the frontier location t increases over time. For each t we compute:

  mean accuracy  A(t)   = mean_k E[acc_k]
  reliability    rho(t) = Var_k(E[acc_k]) / (Var_k(E[acc_k]) + mean_k Var(acc_k | theta_k))
                 i.e. share of observed score variance due to true capability differences; 1 - rho is
                 the share due to response/seed noise (each re-run draws fresh Bernoulli responses).
  Kendall tau    tau(t) = E[tau(observed scores, theta)] estimated with R Monte-Carlo replicates.

Saturation accuracy A* = mean accuracy at which rho (or tau) first falls below threshold as t increases.
Effective lifetime     = length of the t-interval over which rho >= threshold.

This is a psychometric simulation of *item responses*; it is complemented by real-LLM data (Exp 2-4).
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "exp1")
os.makedirs(RES, exist_ok=True)
SEED = 42


# ---------------------------------------------------------------- difficulty distributions
def draw_difficulties(family, n, sd, rng):
    """Draw n item difficulties with standard deviation `sd` (location 0) from a shape family."""
    if family == "point":          # all items (almost) equally hard
        z = rng.normal(0, 0.02, n) / 1.0
        return z  # sd fixed ~0 regardless of `sd`
    if family == "uniform":
        z = rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    elif family == "normal":
        z = rng.normal(0, 1, n)
    elif family == "laplace":
        z = rng.laplace(0, 1 / np.sqrt(2), n)
    elif family == "t3":           # heavy, symmetric tails (variance-standardised)
        z = rng.standard_t(3, n) / np.sqrt(3)
    elif family == "lognormal_hard":  # right-skewed: long tail of very hard items
        x = rng.lognormal(0, 1, n)
        z = (x - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
    elif family == "lognormal_easy":  # left-skewed: long tail of very easy items
        x = rng.lognormal(0, 1, n)
        z = -(x - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
    else:
        raise ValueError(family)
    return z * sd


FAMILIES = ["point", "uniform", "normal", "laplace", "t3", "lognormal_hard", "lognormal_easy"]


# ---------------------------------------------------------------- core computation
def curves(b, a, c, thetas_rel, t_grid, n_mc=0, rng=None):
    """Return DataFrame with A, rho, (tau) for each frontier location t.

    b, a: item params (N,), c: guessing floor, thetas_rel: (K,) cohort offsets.
    Uses exact Poisson-binomial mean/variance; tau via MC with normal approximation of scores.
    """
    out = []
    for t in t_grid:
        th = t + thetas_rel                                  # (K,)
        P = c + (1 - c) / (1 + np.exp(-a[None] * (th[:, None] - b[None])))  # (K, N)
        N = P.shape[1]
        mu = P.mean(1)                                       # expected accuracy per model
        v = (P * (1 - P)).sum(1) / N ** 2                    # response-noise variance of accuracy
        sig = mu.var(ddof=1)
        noise = v.mean()
        rho = sig / (sig + noise) if sig + noise > 0 else 0.0
        row = dict(t=t, acc=mu.mean(), rho=rho, signal_var=sig, noise_var=noise)
        if n_mc:
            obs = mu[None] + np.sqrt(v)[None] * rng.standard_normal((n_mc, len(mu)))
            taus = [stats.kendalltau(o, th).statistic for o in obs]
            row["tau"] = float(np.nanmean(taus))
            # probability that the truly-best model is ranked first
            row["p_top1"] = float(np.mean(obs.argmax(1) == np.argmax(th)))
        out.append(row)
    return pd.DataFrame(out)


def crossing_acc(df, col, thr):
    """Mean accuracy at which `col` first drops below thr after its peak (ceiling side)."""
    i_peak = int(df[col].values.argmax())
    if df[col].values[i_peak] < thr:
        return np.nan  # never discriminates at this threshold
    post = df.iloc[i_peak:]
    below = post[post[col] < thr]
    if len(below) == 0:
        return np.nan
    j = below.index[0]
    # linear interpolation in accuracy between j-1 and j
    k = df.index.get_loc(j)
    r0, r1 = df.iloc[k - 1], df.iloc[k]
    w = (r0[col] - thr) / (r0[col] - r1[col])
    return float(r0.acc + w * (r1.acc - r0.acc))


def lifetime(df, col, thr):
    """Length of the t-interval where col >= thr (grid-resolution)."""
    dt = df.t.diff().median()
    return float((df[col] >= thr).sum() * dt)


def cohort(K, sigma_c):
    return sigma_c * stats.norm.ppf((np.arange(K) + 0.5) / K)


# ---------------------------------------------------------------- experiments
def main():
    rng = np.random.default_rng(SEED)
    t_grid = np.linspace(-8, 14, 221)

    base = dict(N=500, K=20, sigma_c=0.3, sd=1.5, a_sd=0.3, c=0.0)
    # ---- 1a. shape comparison at matched SD, with sensitivity settings
    settings = [("base", {})] + [
        ("N=100", {"N": 100}), ("N=2000", {"N": 2000}),
        ("sigma_c=0.15", {"sigma_c": 0.15}), ("sigma_c=0.6", {"sigma_c": 0.6}),
        ("sd=0.75", {"sd": 0.75}), ("sd=3.0", {"sd": 3.0}),
        ("guess c=0.25", {"c": 0.25}), ("K=10", {"K": 10}), ("a_sd=0", {"a_sd": 0.0}),
    ]
    rows, curve_rows = [], []
    for sname, upd in settings:
        cfg = {**base, **upd}
        th_rel = cohort(cfg["K"], cfg["sigma_c"])
        for fam in FAMILIES:
            for rep in range(20):  # different random item draws
                b = draw_difficulties(fam, cfg["N"], cfg["sd"], rng)
                a = np.exp(rng.normal(0, cfg["a_sd"], cfg["N"])) if cfg["a_sd"] > 0 else np.ones(cfg["N"])
                n_mc = 200 if (sname == "base" and rep < 3) else 0
                df = curves(b, a, cfg["c"], th_rel, t_grid, n_mc=n_mc, rng=rng)
                r = dict(setting=sname, family=fam, rep=rep, var_b=float(b.var()), sd_b=float(b.std()),
                         acc_star_rho50=crossing_acc(df, "rho", 0.5),
                         acc_star_rho70=crossing_acc(df, "rho", 0.7),
                         life_rho50=lifetime(df, "rho", 0.5), life_rho70=lifetime(df, "rho", 0.7),
                         peak_rho=float(df.rho.max()))
                if n_mc:
                    r["acc_star_tau50"] = crossing_acc(df, "tau", 0.5)
                    r["life_tau50"] = lifetime(df, "tau", 0.5)
                rows.append(r)
                if sname == "base" and rep < 3:
                    df["family"], df["rep"] = fam, rep
                    curve_rows.append(df)
        print("done setting", sname, flush=True)
    shape = pd.DataFrame(rows)
    shape.to_csv(os.path.join(RES, "shape_comparison.csv"), index=False)
    pd.concat(curve_rows).to_csv(os.path.join(RES, "base_curves.csv"), index=False)

    # ---- 1b. lifetime vs difficulty variance across random benchmark configurations
    rows = []
    th_rel = cohort(base["K"], base["sigma_c"])
    for i in range(600):
        fam = rng.choice(FAMILIES[1:])
        sd = float(np.exp(rng.uniform(np.log(0.25), np.log(4.0))))
        N = int(rng.choice([100, 250, 500, 1000]))
        b = draw_difficulties(fam, N, sd, rng)
        a = np.exp(rng.normal(0, base["a_sd"], N))
        df = curves(b, a, 0.0, th_rel, t_grid)
        q = np.quantile(b, [0.05, 0.25, 0.75, 0.95])
        rows.append(dict(family=fam, sd_target=sd, N=N, var_b=float(b.var()), sd_b=float(b.std()),
                         iqr_b=float(q[2] - q[1]), range90_b=float(q[3] - q[0]),
                         kurt_b=float(stats.kurtosis(b)), skew_b=float(stats.skew(b)),
                         life_rho50=lifetime(df, "rho", 0.5), life_rho70=lifetime(df, "rho", 0.7),
                         acc_star_rho50=crossing_acc(df, "rho", 0.5)))
    life = pd.DataFrame(rows)
    life.to_csv(os.path.join(RES, "lifetime_configs.csv"), index=False)

    # ---- summaries
    summ = (shape.groupby(["setting", "family"])
            [["acc_star_rho50", "acc_star_rho70", "acc_star_tau50", "life_rho50", "sd_b"]]
            .agg(["mean", "std"]).round(4))
    summ.to_csv(os.path.join(RES, "shape_summary.csv"))
    corr = {}
    for x in ["var_b", "sd_b", "iqr_b", "range90_b"]:
        for y in ["life_rho50", "life_rho70"]:
            r, p = stats.pearsonr(life[x], life[y])
            rs, _ = stats.spearmanr(life[x], life[y])
            corr[f"{x}~{y}"] = dict(pearson=r, p=p, spearman=rs)
    # within N=500 only (removes N as a nuisance factor)
    sub = life[life.N == 500]
    for x in ["var_b", "sd_b", "range90_b"]:
        r, p = stats.pearsonr(sub[x], sub["life_rho50"])
        corr[f"N500:{x}~life_rho50"] = dict(pearson=r, p=p, spearman=stats.spearmanr(sub[x], sub["life_rho50"]).statistic)
    json.dump(corr, open(os.path.join(RES, "lifetime_correlations.json"), "w"), indent=2)
    print(summ.xs("base", level="setting"))
    print(json.dumps(corr, indent=1))


if __name__ == "__main__":
    main()
