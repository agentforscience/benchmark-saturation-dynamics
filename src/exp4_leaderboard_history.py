"""Exp 4: real saturation trajectories on the Open LLM Leaderboard v1 (Jun 2023 - Jun 2024).

For each month-end M, the "frontier cohort" is the top-K (K=20) unique models by leaderboard Average
among all submissions up to M. For each of the 6 benchmarks we compute, within that cohort:
  mean score, SD of scores, binomial (item-sampling) noise variance p(1-p)/N_items, and
  reliability rho = 1 - noise / observed variance  (share of score variance that is not
  attributable to item-sampling noise).  1 - rho = noise share of ranking variance.
Benchmark lifetime (months with rho >= 0.5) is related to the item-difficulty variance from Exp 2.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "exp4")
os.makedirs(RES, exist_ok=True)
N_ITEMS = {"ARC": 1172, "HellaSwag": 10042, "MMLU": 14042, "TruthfulQA": 817, "Winogrande": 1267, "GSM8K": 1319}
K = 20


def main():
    o = pd.read_parquet(os.path.join(ROOT, "datasets", "olb_old", "contents.parquet"))
    o = o.dropna(subset=["Average ⬆️"])
    o = o[~o.fullname.str.contains("contaminat", case=False)]
    o["date"] = pd.to_datetime(o["date"], utc=True, errors="coerce")
    o = o.dropna(subset=["date"]).sort_values("date").drop_duplicates("fullname", keep="first")
    months = pd.date_range("2023-06-30", "2024-06-30", freq="ME", tz="UTC")
    rows = []
    for M in months:
        sub = o[o.date <= M]
        if len(sub) < K:
            continue
        top = sub.nlargest(K, "Average ⬆️")
        for bch, n in N_ITEMS.items():
            p = top[bch].values / 100.0
            v_obs = p.var(ddof=1)
            noise = np.mean(p * (1 - p) / n)
            rows.append(dict(month=M.strftime("%Y-%m"), bench=bch, n_submissions=len(sub), mean=p.mean(),
                             sd=np.sqrt(v_obs), noise_sd=np.sqrt(noise),
                             rho=max(0.0, 1 - noise / v_obs) if v_obs > 0 else 0.0,
                             noise_share=min(1.0, noise / v_obs) if v_obs > 0 else 1.0,
                             top_avg=top["Average ⬆️"].mean()))
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(RES, "history.csv"), index=False)
    piv = H.pivot(index="month", columns="bench", values="mean").round(3)
    print("cohort mean score\n", piv.to_string())
    print("reliability rho\n", H.pivot(index="month", columns="bench", values="rho").round(2).to_string())
    life = H.groupby("bench").apply(lambda g: int((g.rho >= 0.5).sum())).rename("months_rho50")
    json.dump(life.to_dict(), open(os.path.join(RES, "lifetimes.json"), "w"), indent=2)
    print(life)


if __name__ == "__main__":
    main()
