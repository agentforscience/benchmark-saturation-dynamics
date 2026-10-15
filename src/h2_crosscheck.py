"""H2 cross-check: measured seed noise of real LLMs vs real leaderboard frontier spreads.

1. From Exp 3 (real models, 10 option-order seeds), estimate the per-item seed-noise constant
   kappa(acc) = N * Var_seed(accuracy) as a function of a model's accuracy level (for a benchmark
   with N items the seed variance of accuracy is kappa / N).
2. Compare with an idealised binomial benchmark-resampling noise acc(1-acc) (item sampling).
3. Apply to the Open LLM Leaderboard v1 top-20 cohort (Exp 4, final month) for each benchmark:
   predicted seed share = (kappa(acc)/N) / observed score variance of the top-20.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "exp3")
N_ITEMS = {"ARC": 1172, "HellaSwag": 10042, "MMLU": 14042, "TruthfulQA": 817, "Winogrande": 1267, "GSM8K": 1319}


def main():
    import glob
    df = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ROOT, "results", "seed_eval", "*.parquet"))])
    P = np.stack(df.probs.values)
    df["greedy"] = (P.argmax(1) == df.correct_pos.values).astype(float)
    # per-item seed variance of correctness, averaged within accuracy bins (item-level accuracy over seeds)
    g = df.groupby(["model", "bench", "seed"]).greedy.mean().reset_index()
    per = g.groupby(["model", "bench"]).greedy.agg(["mean", "var"]).reset_index()
    per["n"] = df.groupby(["model", "bench"]).qid.nunique().values
    per["kappa"] = per["var"] * per["n"]             # N * Var_seed(acc)
    per["binom"] = per["mean"] * (1 - per["mean"])
    per["ratio_to_binom"] = per["kappa"] / per["binom"]
    per.to_csv(os.path.join(RES, "seed_kappa.csv"), index=False)
    # simple model: kappa = r * acc(1-acc)  (seed noise as a fraction r of binomial noise)
    r_hat = float(np.median(per.ratio_to_binom))
    print(per.round(4).to_string())
    print("median kappa / [acc(1-acc)] =", round(r_hat, 3))

    H = pd.read_csv(os.path.join(ROOT, "results", "exp4", "history.csv"))
    last = H[H.month == H.month.max()]
    rows = []
    for _, row in last.iterrows():
        n = N_ITEMS[row.bench]
        seed_var = r_hat * row["mean"] * (1 - row["mean"]) / n
        obs_var = row.sd ** 2
        rows.append(dict(bench=row.bench, top20_mean=row["mean"], top20_sd=row.sd, n_items=n,
                         pred_seed_sd=np.sqrt(seed_var), pred_seed_share=min(1.0, seed_var / obs_var)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "leaderboard_seed_share.csv"), index=False)
    print(out.round(4).to_string())
    json.dump(dict(r_hat=r_hat), open(os.path.join(RES, "kappa_ratio.json"), "w"))


if __name__ == "__main__":
    main()
