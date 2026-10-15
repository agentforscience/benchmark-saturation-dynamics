"""Build model x item response matrices from the downloaded v1 leaderboard per-item results.

Quality control:
  * keep models with >= 60 of the 62 tasks;
  * per task, keep items answered by >= 95% of the kept models and models covering >= 95% of those
    items (key mismatches from format changes show up as low coverage and are dropped);
  * remaining missing cells (rare) are left as NaN.
Output: results/matrices/<task>.parquet (rows = models), results/matrices/models.csv
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "datasets", "v1_items")
OUT = os.path.join(ROOT, "results", "matrices")
os.makedirs(OUT, exist_ok=True)


def main():
    meta = pd.read_csv(os.path.join(IN, "_models.csv"))
    data = {}
    for f in glob.glob(os.path.join(IN, "*.pkl")):
        name = os.path.basename(f)[:-4].replace("__", "/", 1)
        d = pickle.load(open(f, "rb"))
        if len(d) >= 60:
            data[name] = d
    print(f"models with >=60 tasks: {len(data)} / {len(glob.glob(os.path.join(IN, '*.pkl')))}")
    tasks = sorted({t for d in data.values() for t in d})
    log = []
    for task in tasks:
        names = [n for n in data if task in data[n]]
        counts = pd.Series([k for n in names for k in data[n][task]]).value_counts()
        items = counts[counts >= 0.95 * len(names)].index
        M = pd.DataFrame({n: pd.Series(data[n][task]).reindex(items) for n in names}).T.astype(float)
        cov = M.notna().mean(1)
        M = M[cov >= 0.95]
        M.columns = [str(c) for c in M.columns]
        M.to_parquet(os.path.join(OUT, f"{task}.parquet"))
        log.append(dict(task=task, n_models=len(M), n_items=M.shape[1], missing=float(M.isna().mean().mean()),
                        mean_acc=float(np.nanmean(M.values)), dropped_models=int((cov < 0.95).sum())))
    log = pd.DataFrame(log)
    log.to_csv(os.path.join(OUT, "_task_summary.csv"), index=False)
    print(log.sort_values("mean_acc").to_string())
    meta[meta.fullname.isin(data)].to_csv(os.path.join(OUT, "models.csv"), index=False)


if __name__ == "__main__":
    main()
