"""Exp 3: evaluation-seed variance of real open LLMs (local CPU inference).

For each model, item (ARC-Easy / ARC-Challenge / MMLU samples) and evaluation seed s, we:
  * permute the 4 answer options with an RNG seeded by (seed, item) -- option-order seed,
  * present a 0-shot multiple-choice prompt through the model's chat template,
  * read the next-token probability of each answer letter (A-D) after the assistant prefix.
The stored letter probabilities allow two kinds of seed-dependent scores:
  greedy-under-permutation (prompt/option-order seed) and temperature-1 sampling (sampling seed).

Output: results/seed_eval/<model_tag>.parquet with columns
  model, bench, qid, seed, perm (json), p_correct, probs (4 floats, in presented order), correct_pos
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LETTERS = ["A", "B", "C", "D"]
SYSTEM = "You are a helpful assistant that answers multiple-choice questions."


def build_prompt(tok, question, options):
    body = question.strip() + "\n\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, options))
    body += "\n\nAnswer with only the letter (A, B, C, or D) of the correct option."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    kw = {}
    if "qwen3" in tok.name_or_path.lower():
        kw["enable_thinking"] = False  # Qwen3: disable thinking block
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # templates without system role
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + body}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    return txt + "The answer is "  # assistant prefix; next token should be a letter


def letter_ids(tok):
    """Token ids for 'A'..'D' (first token of the letter, with and without leading space)."""
    ids = []
    for L in LETTERS:
        cands = {tok.encode(L, add_special_tokens=False)[0], tok.encode(" " + L, add_special_tokens=False)[-1]}
        ids.append(sorted(cands))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="items per bench (debug)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    tag = args.model.split("/")[-1]
    out_dir = os.path.join(ROOT, "results", "seed_eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.parquet")

    items = pd.read_json(os.path.join(ROOT, "datasets", "eval_items", "items.jsonl"), lines=True)
    if args.limit:
        items = items.groupby("bench").head(args.limit)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    model.eval()
    lids = letter_ids(tok)

    # build all (item, seed) prompts
    jobs = []
    for _, r in items.iterrows():
        for s in range(args.seeds):
            # seed 0 = original order; other seeds = random permutation keyed on (seed, qid)
            if s == 0:
                perm = list(range(4))
            else:
                rng = np.random.default_rng([s, int(hashlib.md5(r.qid.encode()).hexdigest()[:8], 16)])
                perm = rng.permutation(4).tolist()
            opts = [r.choices[k] for k in perm]
            jobs.append(dict(bench=r.bench, qid=r.qid, seed=s, perm=perm,
                             correct_pos=perm.index(r.answer), prompt=build_prompt(tok, r.question, opts)))
    # sort by length for efficient batching
    order = np.argsort([len(j["prompt"]) for j in jobs])
    t0 = time.time()
    rows = []
    with torch.inference_mode():
        for bi in range(0, len(order), args.batch):
            batch = [jobs[k] for k in order[bi:bi + args.batch]]
            enc = tok([j["prompt"] for j in batch], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, -1)
            # probability mass of each letter (sum over its token variants), renormalised over A-D
            mass = torch.stack([lp[:, ids].exp().sum(-1) for ids in lids], -1)
            probs = (mass / mass.sum(-1, keepdim=True)).cpu().numpy()
            mass = mass.cpu()
            for j, p, m in zip(batch, probs, mass.sum(-1).numpy()):
                rows.append(dict(model=tag, bench=j["bench"], qid=j["qid"], seed=j["seed"], perm=json.dumps(j["perm"]),
                                 correct_pos=j["correct_pos"], probs=p.tolist(), p_correct=float(p[j["correct_pos"]]),
                                 letter_mass=float(m)))
            if (bi // args.batch) % 50 == 0:
                el = time.time() - t0
                print(f"{tag}: {bi + len(batch)}/{len(jobs)} prompts, {el:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    acc = (df.probs.apply(np.argmax) == df.correct_pos).groupby(df.bench).mean()
    print(tag, "done in", round(time.time() - t0), "s; greedy acc:", acc.round(3).to_dict(),
          "mean letter mass", round(df.letter_mass.mean(), 3))


if __name__ == "__main__":
    main()
