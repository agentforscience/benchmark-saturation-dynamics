"""Generate all report figures from results/ into figures/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
R = lambda *p: os.path.join(ROOT, "results", *p)  # noqa: E731

# validated categorical palette (fixed order) + secondary encoding via markers
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
MARK = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 150,
                     "legend.frameon": False})
FAMS = ["point", "uniform", "normal", "laplace", "t3", "lognormal_hard", "lognormal_easy"]


def fig_exp1():
    cur = pd.read_csv(R("exp1", "base_curves.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, f in enumerate(FAMS):
        g = cur[(cur.family == f) & (cur.rep == 0)]
        for ax, col in zip(axes, ["rho", "tau"]):
            gg = g[g.acc > 0.5]
            ax.plot(gg.acc * 100, gg[col], color=PAL[i], label=f, marker=MARK[i], markevery=12, ms=5)
    for ax, lab in zip(axes, ["Reliability ρ (capability share of score variance)", "Kendall τ (observed vs true ranking)"]):
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.axvline(85, color=GRID, lw=1); ax.axvline(95, color=GRID, lw=1)
        ax.set_xlabel("Cohort mean accuracy (%)"); ax.set_ylabel(lab); ax.set_xlim(50, 100)
    axes[0].legend(title="difficulty distribution (SD matched = 1.5)", fontsize=8)
    fig.suptitle("Exp 1 (IRT): discrimination vs. saturation, N=500 items, K=20 models, cohort SD 0.3", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp1_discrimination_vs_accuracy.png")); plt.close(fig)

    s = pd.read_csv(R("exp1", "shape_comparison.csv"))
    settings = ["base", "N=100", "N=2000", "sigma_c=0.15", "sigma_c=0.6", "sd=0.75", "sd=3.0", "guess c=0.25"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(settings)); w = 0.12
    for i, f in enumerate(FAMS):
        for ax, col in zip(axes, ["acc_star_rho50", "life_rho50"]):
            m = [s[(s.setting == st) & (s.family == f)][col].mean() * (100 if "acc" in col else 1) for st in settings]
            ax.bar(x + (i - 3) * w, m, w * 0.85, color=PAL[i], label=f)
    axes[0].set_ylim(75, 100); axes[0].set_ylabel("Saturation accuracy A* (%) where ρ < 0.5")
    axes[1].set_ylabel("Effective lifetime (logits of ability with ρ ≥ 0.5)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(settings, rotation=30, ha="right", fontsize=8)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Exp 1: saturation point and lifetime by difficulty shape across settings", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp1_shape_settings.png")); plt.close(fig)

    L = pd.read_csv(R("exp1", "lifetime_configs.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, (N, g) in zip(axes, L.groupby("N")):
        for i, f in enumerate(FAMS[1:], start=1):
            gg = g[g.family == f]
            ax.scatter(gg.var_b, gg.life_rho50, s=12, color=PAL[i], marker=MARK[i], label=f, alpha=0.8)
        r = np.corrcoef(g.var_b, g.life_rho50)[0, 1]
        ax.set_xscale("log"); ax.set_title(f"N = {N} items   (r = {r:.2f})", fontsize=10)
        ax.set_xlabel("Var(item difficulty) [logit²]")
    axes[0].set_ylabel("Lifetime (logits, ρ ≥ 0.5)"); axes[-1].legend(fontsize=7)
    fig.suptitle("Exp 1: lifetime vs difficulty variance depends on benchmark size", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp1_lifetime_vs_variance.png")); plt.close(fig)


def fig_exp2():
    T = pd.read_csv(R("exp2", "task_stats.csv"))
    C = pd.read_csv(R("exp2", "cohort_curves.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    big = ["arc", "hellaswag", "winogrande", "truthfulqa", "gsm8k"]
    for i, t in enumerate(big):
        g = C[C.task == t]
        axes[0].plot(g.acc * 100, g.rel_sh, color=PAL[i], marker=MARK[i], ms=4, label=t)
    mm = C[C.task.str.startswith("mmlu_")]
    axes[0].scatter(mm.acc * 100, mm.rel_sh, s=4, color="#9e9d98", label="57 MMLU subjects", zorder=0)
    axes[0].axhline(0.5, color=INK2, ls="--", lw=1)
    axes[0].set_xlabel("Cohort mean accuracy (%)"); axes[0].set_ylabel("Split-half reliability (N=90 items, K=20)")
    axes[0].legend(fontsize=8)
    # binned MMLU curve
    bins = np.arange(0.2, 1.0, 0.05)
    mm = mm.assign(bin=pd.cut(mm.acc, bins))
    b = mm.groupby("bin", observed=True).rel_sh.agg(["mean", "count"])
    b = b[b["count"] >= 10]
    axes[1].plot([iv.mid * 100 for iv in b.index], b["mean"], color=PAL[0], marker="o")
    axes[1].axhline(0.5, color=INK2, ls="--", lw=1)
    axes[1].set_xlabel("Cohort mean accuracy (%)"); axes[1].set_ylabel("Mean split-half reliability")
    axes[1].set_title("MMLU subjects, binned (≥10 cohort-task points per bin)", fontsize=10)
    fig.suptitle("Exp 2: real-model discrimination vs saturation (Open LLM Leaderboard v1)", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_reliability_vs_accuracy.png")); plt.close(fig)

    corr = json.load(open(R("exp2", "lifetime_correlations.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y, lab in zip(axes, ["life_emp", "life_model"],
                          ["Empirical lifetime (fraction of cohorts with ρ ≥ 0.5)", "Model-based lifetime (θ-SD units)"]):
        isb = T.bench != "mmlu"
        ax.scatter(T.var_b[~isb], T[y][~isb], s=18, color=PAL[0], label="MMLU subjects")
        ax.scatter(T.var_b[isb], T[y][isb], s=40, color=PAL[1], marker="s", label="ARC/HS/WG/TQA/GSM8K")
        for _, r in T[isb].iterrows():
            ax.annotate(r.task, (r.var_b, r[y]), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
        c = corr[f"var_b~{y}"]
        ax.set_title(f"Pearson r = {c['pearson']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]", fontsize=10)
        ax.set_xlabel("Var(item difficulty), Rasch logit²"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=8)
    fig.suptitle("Exp 2: lifetime vs item-difficulty variance across 62 real tasks", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_lifetime_vs_variance.png")); plt.close(fig)

    if os.path.exists(R("exp2", "resampled_benchmarks.csv")):
        S = pd.read_csv(R("exp2", "resampled_benchmarks.csv"))
        fams = ["uniform", "normal", "laplace", "t3", "lognormal_hard"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, (pool, g) in zip(axes, S.groupby("pool")):
            for i, f in enumerate(fams):
                gg = g[g.family == f]
                gg = gg.assign(bin=pd.cut(gg.acc, np.arange(0.3, 1.001, 0.05)))
                b = gg.groupby("bin", observed=True).rel_sh.agg(["mean", "count"])
                b = b[b["count"] >= 5]
                ax.plot([iv.mid * 100 for iv in b.index], b["mean"], color=PAL[i + 1], marker=MARK[i + 1], ms=5, label=f)
            ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.set_title(f"{pool} item pool (N=200 items)", fontsize=10)
            ax.set_xlabel("Cohort mean accuracy (%)")
        axes[0].set_ylabel("Split-half reliability (K=20 models)"); axes[0].legend(fontsize=8)
        fig.suptitle("Exp 2e: real items resampled to prescribed difficulty shapes (SD = 1 logit)", color=INK)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp2_resampled_shapes.png")); plt.close(fig)


def fig_exp3():
    st = pd.read_csv(R("exp3", "stratified.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, b in enumerate(["mmlu", "arc_challenge", "arc_easy", "pooled"]):
        g = st[(st.bench == b) & (st.score == "greedy")].sort_values("mean_acc")
        axes[0].plot(g.mean_acc * 100, g.seed_share, color=PAL[i], marker=MARK[i], ms=5, label=b)
        g = st[(st.bench == b) & (st.score == "sampled")].sort_values("mean_acc")
        axes[1].plot(g.mean_acc * 100, g.seed_share, color=PAL[i], marker=MARK[i], ms=5, label=b)
    for ax, t in zip(axes, ["Option-order seeds, greedy answer", "Option-order + sampling (T=1) seeds"]):
        ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Cohort mean accuracy on 150-item subset (%)"); ax.set_title(t, fontsize=10)
    axes[0].set_ylabel("Seed share of score variance"); axes[0].legend(fontsize=8)
    fig.suptitle("Exp 3: real LLMs (stronger half of models), difficulty-stratified item subsets", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp3_seed_share_vs_accuracy.png")); plt.close(fig)


def fig_exp4():
    H = pd.read_csv(R("exp4", "history.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (b, g) in enumerate(H.groupby("bench")):
        axes[0].plot(g.month, g["mean"] * 100, color=PAL[i], marker=MARK[i], ms=5, label=b)
        axes[1].plot(g.month, g.rho, color=PAL[i], marker=MARK[i], ms=5, label=b)
    axes[0].set_ylabel("Top-20 cohort mean score (%)"); axes[1].set_ylabel("Reliability ρ vs item-sampling noise")
    axes[1].axhline(0.5, color=INK2, ls="--", lw=1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Exp 4: Open LLM Leaderboard v1 frontier cohort over time", color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "exp4_leaderboard_history.png")); plt.close(fig)


if __name__ == "__main__":
    for f in [fig_exp1, fig_exp2, fig_exp3, fig_exp4]:
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            print("FAILED", f.__name__, repr(e))
