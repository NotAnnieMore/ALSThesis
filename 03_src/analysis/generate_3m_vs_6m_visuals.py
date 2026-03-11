"""
Generate comparative 3m vs 6m visual artefacts for the Overleaf thesis.

Outputs
-------
Figures:
  04_outputs/figures/step5v2_3m_prauc_barchart.{pdf,png}
  04_outputs/figures/step5v2_3m_vs_6m_prauc.{pdf,png}
  04_outputs/figures/step5v2_3m_vs_6m_scatter.{pdf,png}

Tables (LaTeX):
  04_outputs/tables/step5v2_tuning_comparison_3m.tex
  04_outputs/tables/step5v2_3m_vs_6m_comparison.tex
"""

from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── paths ──
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR_3M = os.path.join(ROOT, "04_outputs", "figures", "step5_3m_vs_6m")
FIG_DIR_TUNING = os.path.join(ROOT, "04_outputs", "figures", "step5_tuning")
TAB_DIR_TUNING = os.path.join(ROOT, "04_outputs", "tables", "step5_tuning")
TAB_DIR_COMP   = os.path.join(ROOT, "04_outputs", "tables", "step5_comparison")
OVERLEAF_FIG = os.path.join(ROOT, "overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join(ROOT, "overleaf", "images", "tables")
for d in [FIG_DIR_3M, FIG_DIR_TUNING, TAB_DIR_TUNING, TAB_DIR_COMP, OVERLEAF_FIG, OVERLEAF_TAB]:
    os.makedirs(d, exist_ok=True)

SUMMARY_3M = os.path.join(TAB_DIR_TUNING, "step5_tuning_summary_3m.csv")
SUMMARY_6M = os.path.join(TAB_DIR_TUNING, "step5_tuning_summary_6m.csv")

t_start = time.time()

df3 = pd.read_csv(SUMMARY_3M)
df6 = pd.read_csv(SUMMARY_6M)

df3["label"] = df3["model"] + "\n" + df3["mode"]
df6["label"] = df6["model"] + "\n" + df6["mode"]

print(f"Loaded 3m: {len(df3)} rows,  6m: {len(df6)} rows")


# ════════════════════════════════════════════════════════════
# 1.  PR-AUC bar chart — 3-month horizon
# ════════════════════════════════════════════════════════════
df3s = df3.sort_values("pr_auc_mean", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#3274a1" if m == "unbalanced" else "#e1812c" for m in df3s["mode"]]
bars = ax.barh(df3s["label"], df3s["pr_auc_mean"], xerr=df3s["pr_auc_std"],
               color=colors, edgecolor="white", capsize=3, height=0.7)

# Annotate each bar with its numeric value
for bar, val in zip(bars, df3s["pr_auc_mean"]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

ax.set_xlabel("PR-AUC (mean ± std, 5-fold GroupKFold)", fontsize=11)
ax.set_title("Optuna-Tuned Models — 3-Month Horizon (DEV, n=1925)", fontsize=13, fontweight="bold")
best_val = df3s["pr_auc_mean"].max()
ax.axvline(x=best_val, ls="--", color="grey", alpha=0.5, lw=0.8)
ax.legend(handles=[Patch(color="#3274a1", label="Unbalanced"),
                   Patch(color="#e1812c", label="Balanced")],
          loc="lower right", fontsize=9)
ax.set_xlim(0.30, 0.50)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_prauc_barchart.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_prauc_barchart.png"), dpi=150)
plt.close(fig)
print("  [OK] 3m PR-AUC bar chart saved")


# ════════════════════════════════════════════════════════════
# 2.  Side-by-side grouped bar chart: 3m vs 6m (matched models)
# ════════════════════════════════════════════════════════════
# Match models that appear in both horizons
df3_key = df3[["model", "mode", "pr_auc_mean", "pr_auc_std"]].copy()
df6_key = df6[["model", "mode", "pr_auc_mean", "pr_auc_std"]].copy()
merged = df3_key.merge(df6_key, on=["model", "mode"], suffixes=("_3m", "_6m"))
merged["label"] = merged["model"] + "\n" + merged["mode"]
merged = merged.sort_values("pr_auc_mean_6m", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(merged))
bar_h = 0.35

bars_6m = ax.barh(y + bar_h / 2, merged["pr_auc_mean_6m"], bar_h,
                  xerr=merged["pr_auc_std_6m"], color="#3274a1",
                  edgecolor="white", capsize=2, label="6-month")
bars_3m = ax.barh(y - bar_h / 2, merged["pr_auc_mean_3m"], bar_h,
                  xerr=merged["pr_auc_std_3m"], color="#e1812c",
                  edgecolor="white", capsize=2, label="3-month")

# Annotate bars
for bar, val in zip(bars_6m, merged["pr_auc_mean_6m"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold",
            color="#3274a1")
for bar, val in zip(bars_3m, merged["pr_auc_mean_3m"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold",
            color="#e1812c")

ax.set_yticks(y)
ax.set_yticklabels(merged["label"], fontsize=9)
ax.set_xlabel("PR-AUC (mean ± std)", fontsize=11)
ax.set_title("PR-AUC Comparison: 3-Month vs 6-Month Horizons (DEV)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim(0.30, 0.50)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_prauc.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_prauc.png"), dpi=150)
plt.close(fig)
print("  [OK] 3m vs 6m grouped bar chart saved")


# ════════════════════════════════════════════════════════════
# 3.  Scatter plot: 3m PR-AUC vs 6m PR-AUC per model×mode
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))

# Diagonal reference line
lims = [0.38, 0.45]
ax.plot(lims, lims, "--", color="grey", alpha=0.5, lw=1, label="y = x (parity)")

# Color by mode
for _, row in merged.iterrows():
    c = "#3274a1" if row["mode"] == "unbalanced" else "#e1812c"
    ax.scatter(row["pr_auc_mean_6m"], row["pr_auc_mean_3m"], c=c,
               s=100, edgecolors="black", linewidths=0.5, zorder=5)
    # Label with model name
    offset_x, offset_y = 0.002, 0.002
    ax.annotate(row["model"], (row["pr_auc_mean_6m"], row["pr_auc_mean_3m"]),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                color=c, fontweight="bold")

# Delta annotations: show delta for each point
for _, row in merged.iterrows():
    delta = row["pr_auc_mean_3m"] - row["pr_auc_mean_6m"]
    side = "above" if delta >= 0 else "below"
    sign = "+" if delta >= 0 else ""
    ax.annotate(f"{sign}{delta:.3f}",
                (row["pr_auc_mean_6m"], row["pr_auc_mean_3m"]),
                textcoords="offset points",
                xytext=(6, -12 if side == "below" else -12),
                fontsize=6.5, color="dimgrey", style="italic")

ax.set_xlabel("PR-AUC — 6-month (DEV)", fontsize=11)
ax.set_ylabel("PR-AUC — 3-month (DEV)", fontsize=11)
ax.set_title("3m vs 6m PR-AUC per Model (Optuna-Tuned)", fontsize=13, fontweight="bold")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="#3274a1", ls="", markersize=8, label="Unbalanced"),
    plt.Line2D([0], [0], marker="o", color="#e1812c", ls="", markersize=8, label="Balanced"),
    plt.Line2D([0], [0], ls="--", color="grey", label="Parity line"),
], loc="upper left", fontsize=9)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_scatter.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR_3M, "step5v2_3m_vs_6m_scatter.png"), dpi=150)
plt.close(fig)
print("  [OK] 3m vs 6m scatter plot saved")


# ════════════════════════════════════════════════════════════
# 4.  LaTeX table — 3m tuning comparison
# ════════════════════════════════════════════════════════════
df3_sorted = df3.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

best_pr  = df3_sorted["pr_auc_mean"].max()
best_roc = df3_sorted["roc_auc_mean"].max()
best_f1  = df3_sorted["f1_mean"].max()
best_f2  = df3_sorted["f2_mean"].max()

def fmt_bold(val, best, std=None):
    s = f"{val:.3f}"
    if std is not None:
        s += f" $\\pm$ {std:.3f}"
    if abs(val - best) < 1e-6:
        return r"\textbf{" + s + "}"
    return s

lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{Optuna-tuned model comparison (3-month horizon, DEV set, GroupKFold $K\!=\!5$, 60 trials). "
    r"Models ranked by PR-AUC. Best value per metric in \textbf{bold}.}",
    r"\label{tab:step5v2_tuning_3m}",
    r"\small",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Model & Mode & PR-AUC & ROC-AUC & F1\textsubscript{@0.5} & F2\textsubscript{@0.5} \\",
    r"\midrule",
]

for _, row in df3_sorted.iterrows():
    model = row["model"].replace("_", r"\_")
    mode  = row["mode"]
    pr    = fmt_bold(row["pr_auc_mean"], best_pr, row["pr_auc_std"])
    roc   = fmt_bold(row["roc_auc_mean"], best_roc)
    f1    = fmt_bold(row["f1_mean"], best_f1)
    f2    = fmt_bold(row["f2_mean"], best_f2)
    lines.append(f"{model} & {mode} & {pr} & {roc} & {f1} & {f2} \\\\")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex_path = os.path.join(TAB_DIR_TUNING, "step5v2_tuning_comparison_3m.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  [OK] 3m LaTeX tuning table saved → {tex_path}")


# ════════════════════════════════════════════════════════════
# 5.  LaTeX table — 3m vs 6m side-by-side for matched models
# ════════════════════════════════════════════════════════════
merged_sorted = merged.sort_values("pr_auc_mean_6m", ascending=False).reset_index(drop=True)
merged_sorted["delta"] = merged_sorted["pr_auc_mean_3m"] - merged_sorted["pr_auc_mean_6m"]

lines2 = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{PR-AUC comparison across prediction horizons for matched model configurations. "
    r"$\Delta$ = 3m $-$ 6m; positive values indicate better performance at 3 months.}",
    r"\label{tab:3m_vs_6m}",
    r"\small",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Model & Mode & PR-AUC\textsubscript{6m} & PR-AUC\textsubscript{3m} & $\Delta$ \\",
    r"\midrule",
]

for _, row in merged_sorted.iterrows():
    model = row["model"].replace("_", r"\_")
    mode  = row["mode"]
    v6    = f"{row['pr_auc_mean_6m']:.3f} $\\pm$ {row['pr_auc_std_6m']:.3f}"
    v3    = f"{row['pr_auc_mean_3m']:.3f} $\\pm$ {row['pr_auc_std_3m']:.3f}"
    d     = row["delta"]
    sign  = "+" if d >= 0 else ""
    ds    = f"{sign}{d:.3f}"
    lines2.append(f"{model} & {mode} & {v6} & {v3} & {ds} \\\\")

lines2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex2_path = os.path.join(TAB_DIR_COMP, "step5v2_3m_vs_6m_comparison.tex")
with open(tex2_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines2))
print(f"  [OK] 3m vs 6m LaTeX comparison table saved → {tex2_path}")


# ════════════════════════════════════════════════════════════
# 6.  Also regenerate 6m bar chart with numeric annotations
# ════════════════════════════════════════════════════════════
df6s = df6.sort_values("pr_auc_mean", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#3274a1" if m == "unbalanced" else "#e1812c" for m in df6s["mode"]]
bars = ax.barh(df6s["label"], df6s["pr_auc_mean"], xerr=df6s["pr_auc_std"],
               color=colors, edgecolor="white", capsize=3, height=0.7)

for bar, val in zip(bars, df6s["pr_auc_mean"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, fontweight="bold")

ax.set_xlabel("PR-AUC (mean ± std, 5-fold GroupKFold)", fontsize=11)
ax.set_title("Optuna-Tuned Models — 6-Month Horizon (DEV, n=1113)", fontsize=13, fontweight="bold")
ax.axvline(x=df6s["pr_auc_mean"].max(), ls="--", color="grey", alpha=0.5, lw=0.8)
ax.legend(handles=[Patch(color="#3274a1", label="Unbalanced"),
                   Patch(color="#e1812c", label="Balanced")],
          loc="lower right", fontsize=9)
ax.set_xlim(0.30, 0.50)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR_TUNING, "step5v2_prauc_barchart.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR_TUNING, "step5v2_prauc_barchart.png"), dpi=150)
plt.close(fig)
print("  [OK] 6m PR-AUC bar chart regenerated with numeric annotations")


# ════════════════════════════════════════════════════════════
# Copy to overleaf
# ════════════════════════════════════════════════════════════
import shutil
for d in [FIG_DIR_3M, FIG_DIR_TUNING]:
    for f in os.listdir(d):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(d, f), os.path.join(OVERLEAF_FIG, f))
for d in [TAB_DIR_TUNING, TAB_DIR_COMP]:
    for f in os.listdir(d):
        if f.endswith(".tex"):
            shutil.copy2(os.path.join(d, f), os.path.join(OVERLEAF_TAB, f))
print("  → Copied figures/tables to overleaf/images/")


# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  All 3m vs 6m artefacts generated in {elapsed:.1f}s")
print(f"{'='*60}")

# Quick comparison summary
print("\n── Top models per horizon ──")
print("6-month (DEV n=1113):")
top6 = df6.sort_values("pr_auc_mean", ascending=False).head(4)
for _, r in top6.iterrows():
    print(f"  {r['model']:20s} {r['mode']:12s}  PR-AUC={r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f}  ROC-AUC={r['roc_auc_mean']:.3f}")

print("\n3-month (DEV n=1925):")
top3 = df3.sort_values("pr_auc_mean", ascending=False).head(4)
for _, r in top3.iterrows():
    print(f"  {r['model']:20s} {r['mode']:12s}  PR-AUC={r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f}  ROC-AUC={r['roc_auc_mean']:.3f}")

print("\n── Deltas (3m - 6m) for matched configs ──")
for _, r in merged_sorted.iterrows():
    d = r["delta"]
    sign = "+" if d >= 0 else ""
    print(f"  {r['model']:20s} {r['mode']:12s}  Δ={sign}{d:.3f}")
