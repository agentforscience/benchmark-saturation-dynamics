"""Download per-item correctness for a stratified sample of Open LLM Leaderboard v1 models.

Source: public HF datasets `open-llm-leaderboard-old/details_<org>__<model>` (lighteval/harness
per-sample outputs). For each model and each of the 62 v1 tasks (ARC-Challenge, HellaSwag,
TruthfulQA-mc1, Winogrande, GSM8K, 57 MMLU subjects) we read only the correctness column and a
text column (hashed into a stable item key) using column-pruned parquet reads.

Output: datasets/v1_items/<eval_name>.pkl  -> {task: {item_key: 0/1}}
        datasets/v1_items/_models.csv      -> selected models with leaderboard metadata
"""
import hashlib
import io
import re
import urllib.parse
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets", "v1_items")
os.makedirs(OUT, exist_ok=True)
SEED = 42
N_BINS, PER_BIN = 20, 11

# task-config prefix -> (task name, correctness column, text column)
TASKS = {
    "harness|arc:challenge|25": ("arc", "acc_norm", "query"),
    "harness|hellaswag|10": ("hellaswag", "acc_norm", "query"),
    "harness|truthfulqa:mc|0": ("truthfulqa", "mc1", "question"),
    "harness|winogrande|5": ("winogrande", "acc", "sentence"),
    "harness|gsm8k|5": ("gsm8k", "acc", "question"),
}
MMLU_PREFIX = "harness|hendrycksTest-"

api, fs = HfApi(), HfFileSystem()
import requests
SESSION = requests.Session()


def norm_key(text):
    """Stable item key: normalized tail of the question text (robust to prefix/format changes)."""
    t = re.sub(r"\s+", " ", str(text)).strip().lower()
    t = re.sub(r"^(question:|q:)\s*", "", t)
    return hashlib.md5(t[-300:].encode()).hexdigest()[:16]


def select_models():
    o = pd.read_parquet(os.path.join(ROOT, "datasets", "olb_old", "contents.parquet"))
    o = o.dropna(subset=["Average ⬆️"])
    o = o[~o.fullname.str.contains("contaminat", case=False)]
    o = o.sort_values("date").drop_duplicates("fullname", keep="last")
    o["bin"] = pd.qcut(o["Average ⬆️"], N_BINS, labels=False)
    sel = o.groupby("bin", group_keys=False).apply(
        lambda g: g.sample(min(PER_BIN, len(g)), random_state=SEED))
    # add the top of the leaderboard (saturation regime), excluding already-selected models
    top = o[~o.fullname.isin(sel.fullname)].nlargest(25, "Average ⬆️")
    return pd.concat([sel, top]).reset_index(drop=True)


def task_of(fname):
    """Map a details filename to (task, metric, textcol, timestamp) or None (non-standard config)."""
    base = fname.split("/")[-1]
    if not base.startswith("details_"):
        return None
    cfg, ts = base[len("details_"):].rsplit("_", 1)
    ts = ts.replace(".parquet", "")
    if cfg in TASKS:
        return (*TASKS[cfg], ts)
    if cfg.startswith(MMLU_PREFIX) and cfg.endswith("|5"):
        subj = cfg[len(MMLU_PREFIX):-2]
        return (f"mmlu_{subj}", "acc", "query", ts)
    return None


def fetch_model(fullname):
    repo = "open-llm-leaderboard-old/details_" + fullname.replace("/", "__")
    out_path = os.path.join(OUT, fullname.replace("/", "__") + ".pkl")
    if os.path.exists(out_path):
        return fullname, "cached"
    for attempt in range(5):
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
            break
        except Exception as e:  # rate limit / transient
            if "404" in str(e) or "not found" in str(e).lower():
                return fullname, "missing"
            time.sleep(10 * (attempt + 1))
    else:
        return fullname, "list_failed"
    latest = {}
    for f in files:
        t = task_of(f)
        if t is None:
            continue
        task, metric, textcol, ts = t
        if task not in latest or ts > latest[task][3]:
            latest[task] = (f, metric, textcol, ts)
    def one(item):
        task, (f, metric, textcol, ts) = item
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/" + urllib.parse.quote(f)
        err = None
        for attempt in range(6):
            try:
                if task == "hellaswag":  # 60MB file: column-pruned remote read
                    fh = fs.open(f"datasets/{repo}/{f}", "rb", block_size=2 ** 20)
                else:
                    r = SESSION.get(url, timeout=120)
                    if r.status_code == 429:
                        raise RuntimeError("429")
                    r.raise_for_status()
                    fh = io.BytesIO(r.content)
                names = pq.ParquetFile(fh).schema_arrow.names
                if metric in names:  # old lighteval format
                    tcol = textcol if textcol in names else "example"
                    tab = pq.read_table(fh, columns=[metric, tcol]).to_pandas()
                    vals = tab[metric].map(lambda v: float(v in (True, "True")) if isinstance(v, (bool, str)) else float(v)).values
                else:  # newer format: metrics struct + 'example'
                    tcol = "example"
                    tab = pq.read_table(fh, columns=["metrics", tcol]).to_pandas()
                    vals = np.array([float(m[metric]) for m in tab["metrics"]])
                keys = [norm_key(x) for x in tab[tcol]]
                return task, dict(zip(keys, (vals > 0.5).astype(np.int8)))
            except Exception as e:
                err = e
                time.sleep(5 * (attempt + 1))
        print(f"  fail {fullname} {task}: {err}", file=sys.stderr)
        return task, None

    res = {}
    with ThreadPoolExecutor(6) as ex:
        for task, d in ex.map(one, latest.items()):
            if d is not None:
                res[task] = d
    with open(out_path, "wb") as fh:
        pickle.dump(res, fh)
    return fullname, f"ok {len(res)} tasks"


if __name__ == "__main__":
    random.seed(SEED)
    sel = select_models()
    sel.drop(columns=["Model"]).to_csv(os.path.join(OUT, "_models.csv"), index=False)
    print(f"selected {len(sel)} models; Average range {sel['Average ⬆️'].min():.1f}-{sel['Average ⬆️'].max():.1f}")
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(fetch_model, n) for n in sel.fullname]
        for i, fu in enumerate(as_completed(futs)):
            name, status = fu.result()
            print(f"[{i+1}/{len(sel)}] {name}: {status}", flush=True)
