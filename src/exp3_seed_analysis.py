"""Exp 3 analysis: does evaluation-seed noise dominate ranking variance at saturation? (H2)

Input: results/seed_eval/*.parquet (real open LLMs, 10 option-order seeds x 1500 items).
Per (model, item, seed) we have the model's probability over the 4 presented options.

Seed-dependent accuracy definitions:
  greedy  : correct iff argmax option == gold under the seed's option permutation (prompt/order seed)
  sampled : answer sampled from the option distribution (T=1) with RNG seed s, under permutation s
For a cohort of models on an item set, X[m, s] = accuracy of model m under seed s:
  seed variance      = mean_m Var_s(X[m, :])
  capability variance= Var_m(mean_s X) - seed variance / S       (unbiased, floored at 0)
  seed share         = seed var / (seed var + capability var)     (> 0.5 => seed noise dominates)
Ranking stability: mean Kendall tau between single-seed rankings and the seed-averaged ranking.

Saturation is varied in two ways: (i) natural benchmarks (MMLU < ARC-C < ARC-E), (ii) difficulty-
stratified item subsets (difficulty estimated on a disjoint set of *weaker* models to avoid selecting
on the cohort's own noise) giving a range of cohort mean accuracies.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "exp3")
os.makedirs(RES, exist_ok=True)
SEED = 42


def load():
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ROOT, "results", "seed_eval", "*.parquet"))])
    rng = np.random.default_rng(SEED)
    P = np.stack(df.probs.values)
    df["greedy"] = (P.argmax(1) == df.correct_pos.values).astype(float)
    # sampled answer (T=1); RNG stream fixed by global seed -> one draw per (model,item,seed)
    u = rng.random(len(df))
    cum = P.cumsum(1)
    df["sampled"] = ((u[:, None] > cum).sum(1).clip(0, 3) == df.correct_pos.values).astype(float)
    df["expected"] = df.p_correct
    return df


def decompose(X):
    """X: (models, seeds) accuracies. Returns dict with variance components & ranking stats."""
    M, S = X.shape
    seed_var = X.var(1, ddof=1).mean()
    cap_var = max(0.0, X.mean(1).var(ddof=1) - seed_var / S)
    share = seed_var / (seed_var + cap_var) if seed_var + cap_var > 0 else np.nan
    mean_rank = X.mean(1)
    taus = [stats.kendalltau(X[:, s], mean_rank).statistic for s in range(S)]
    # pairwise flip prob between two independent single-seed evaluations
    flips = []
    for s1 in range(S):
        for s2 in range(s1 + 1, S):
            d1 = np.sign(X[:, s1][:, None] - X[:, s1][None])
            d2 = np.sign(X[:, s2][:, None] - X[:, s2][None])
            iu = np.triu_indices(M, 1)
            flips.append(np.mean(d1[iu] * d2[iu] < 0))
    top1 = np.mean([np.argmax(X[:, s]) == np.argmax(mean_rank) for s in range(S)])
    return dict(seed_var=seed_var, cap_var=cap_var, seed_share=share, tau_single_vs_mean=np.nanmean(taus),
                pair_flip=np.mean(flips), top1_agree=top1, mean_acc=X.mean(), n_models=M)


def matrix(df, models, qids, col):
    sub = df[df.model.isin(models) & df.qid.isin(qids)]
    return sub.pivot_table(index="model", columns="seed", values=col, aggfunc="mean").loc[models].values


def main():
    df = load()
    models = sorted(df.model.unique())
    acc = df.groupby(["model", "bench"]).expected.mean().unstack()
    acc_g = df.groupby(["model", "bench"]).greedy.mean().unstack()
    print("models:", len(models))
    print("greedy accuracy (mean over seeds):\n", acc_g.round(3).to_string())
    acc_g.to_csv(os.path.join(RES, "model_accuracy_greedy.csv"))
    # per-model seed SD of accuracy (greedy & sampled)
    sd = (df.groupby(["model", "bench", "seed"])[["greedy", "sampled"]].mean()
          .groupby(["model", "bench"]).std().unstack())
    sd.to_csv(os.path.join(RES, "seed_sd.csv"))

    rows, size_rows = [], []
    rng0 = np.random.default_rng(SEED + 1)
    # (i) natural benchmarks; cohorts = all models, and frontier cohort = top-6 by greedy accuracy
    for bench in ["mmlu", "arc_challenge", "arc_easy"]:
        qids = df[df.bench == bench].qid.unique()
        ranked = acc_g[bench].sort_values().index.tolist()
        for cname, cm in [("all", ranked), ("top6", ranked[-6:]), ("top10", ranked[-10:])]:
            for col in ["greedy", "sampled"]:
                # per-item x seed array for item-bootstrap CIs
                sub = df[df.model.isin(cm) & df.qid.isin(qids)]
                A = sub.pivot_table(index=["model", "qid"], columns="seed", values=col).loc[cm]
                A3 = A.values.reshape(len(cm), len(qids), -1)              # (models, items, seeds)
                d = decompose(A3.mean(1))
                bs = []
                for _ in range(300):
                    ii = rng0.integers(0, len(qids), len(qids))
                    bs.append(decompose(A3[:, ii].mean(1))["seed_share"])
                rows.append(dict(analysis="natural", bench=bench, cohort=cname, score=col, n_items=len(qids), **d,
                                 share_ci_lo=float(np.nanpercentile(bs, 2.5)), share_ci_hi=float(np.nanpercentile(bs, 97.5))))
                # dependence on benchmark size: subsample items (seed variance ~ 1/N)
                if cname == "top6" and col == "greedy":
                    for n in [50, 100, 200, 500]:
                        sh = [decompose(A3[:, rng0.choice(len(qids), n, replace=False)].mean(1))["seed_share"] for _ in range(100)]
                        size_rows.append(dict(bench=bench, n_items=n, seed_share=float(np.nanmean(sh))))
    nat = pd.DataFrame(rows)
    nat.to_csv(os.path.join(RES, "natural_benchmarks.csv"), index=False)
    pd.DataFrame(size_rows).to_csv(os.path.join(RES, "seed_share_vs_nitems.csv"), index=False)
    print(pd.DataFrame(size_rows).pivot(index="n_items", columns="bench", values="seed_share").round(3))
    print(nat[["bench", "cohort", "score", "mean_acc", "seed_share", "tau_single_vs_mean", "pair_flip", "top1_agree", "share_ci_lo", "share_ci_hi"]].round(3).to_string())

    # (ii) difficulty-stratified subsets: difficulty from the weaker half, cohort = stronger half
    rng = np.random.default_rng(SEED)
    overall = acc_g.mean(1).sort_values()
    weak, strong = overall.index[: len(overall) // 2].tolist(), overall.index[len(overall) // 2:].tolist()
    item_p = df[df.model.isin(weak)].groupby("qid").expected.mean()
    rows = []
    for bench in ["mmlu", "arc_challenge", "arc_easy", "pooled"]:
        q = item_p[df.drop_duplicates("qid").set_index("qid").loc[item_p.index].bench.eq(bench).values] if bench != "pooled" else item_p
        q = q.sort_values()
        n = 150
        # sliding windows of n items from hardest to easiest
        for start in np.linspace(0, len(q) - n, 15).astype(int):
            qids = q.index[start:start + n]
            for col in ["greedy", "sampled"]:
                X = matrix(df, strong, qids, col)
                d = decompose(X)
                # bootstrap over items for CI of seed share
                rows.append(dict(analysis="stratified", bench=bench, start=int(start), score=col, n_items=n, **d))
    strat = pd.DataFrame(rows)
    strat.to_csv(os.path.join(RES, "stratified.csv"), index=False)
    print(strat[strat.score == "greedy"][["bench", "mean_acc", "seed_share", "tau_single_vs_mean", "pair_flip"]].round(3).to_string())

    # summary tests: is seed share > 0.5 when cohort mean accuracy >= 0.9 ?
    summ = {}
    for col in ["greedy", "sampled"]:
        s = strat[(strat.score == col) & (strat.bench == "pooled")]
        hi, lo = s[s.mean_acc >= 0.9], s[s.mean_acc < 0.8]
        summ[col] = dict(
            share_hi_mean=float(hi.seed_share.mean()) if len(hi) else None, n_hi=len(hi),
            share_lo_mean=float(lo.seed_share.mean()) if len(lo) else None, n_lo=len(lo),
            spearman_acc_share=float(stats.spearmanr(s.mean_acc, s.seed_share).statistic),
            spearman_p=float(stats.spearmanr(s.mean_acc, s.seed_share).pvalue))
    json.dump(summ, open(os.path.join(RES, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
