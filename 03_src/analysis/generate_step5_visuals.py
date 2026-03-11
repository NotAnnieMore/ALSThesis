"""
Generate all Step-5 visual artefacts for the Overleaf thesis.

Outputs
-------
Figures:
  04_outputs/figures/step5v2_prauc_barchart.pdf
  04_outputs/figures/step5v2_balanced_vs_unbalanced.pdf
  04_outputs/figures/step5v2_metric_heatmap.pdf
  04_outputs/figures/step5v2_fvc_missingness_forest.pdf

Tables (LaTeX):
  04_outputs/tables/step5v2_tuning_comparison.tex
  04_outputs/tables/step5v2_fvc_missingness.tex

Analysis:
  04_outputs/tables/step5v2_metric_rank_correlation.csv
"""

from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.stats import spearmanr

# ── paths ──
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR    = os.path.join(ROOT, "04_outputs", "figures", "step5_tuning")
TAB_DIR    = os.path.join(ROOT, "04_outputs", "tables", "step5_tuning")
OVERLEAF_FIG = os.path.join(ROOT, "overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join(ROOT, "overleaf", "images", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OVERLEAF_FIG, exist_ok=True)
os.makedirs(OVERLEAF_TAB, exist_ok=True)

SUMMARY_6M        = os.path.join(TAB_DIR, "step5_tuning_summary_6m.csv")
FVC_MISSINGNESS   = os.path.join(ROOT, "04_outputs", "tables", "step4", "step4_fvc_missingness_analysis.csv")

t_start = time.time()

# ════════════════════════════════════════════════════════════
# 1.  Load tuning summary
# ════════════════════════════════════════════════════════════
df = pd.read_csv(SUMMARY_6M)
df["label"] = df["model"] + "\n" + df["mode"]
df = df.sort_values("pr_auc_mean", ascending=True).reset_index(drop=True)  # low→high for horizontal bar

print(f"Loaded tuning summary: {len(df)} rows")

# ════════════════════════════════════════════════════════════
# 2.  PR-AUC horizontal bar chart
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#3274a1" if m == "unbalanced" else "#e1812c" for m in df["mode"]]
bars = ax.barh(df["label"], df["pr_auc_mean"], xerr=df["pr_auc_std"],
               color=colors, edgecolor="white", capsize=3, height=0.7)
ax.set_xlabel("PR-AUC (mean ± std, 5-fold GroupKFold)", fontsize=11)
ax.set_title("Optuna-Tuned Models — 6-Month Horizon (DEV)", fontsize=13, fontweight="bold")
ax.axvline(x=df["pr_auc_mean"].max(), ls="--", color="grey", alpha=0.5, lw=0.8)

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#3274a1", label="Unbalanced"),
                   Patch(color="#e1812c", label="Balanced")],
          loc="lower right", fontsize=9)
ax.set_xlim(0.30, 0.50)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "step5v2_prauc_barchart.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "step5v2_prauc_barchart.png"), dpi=150)
plt.close(fig)
print("  [OK] PR-AUC bar chart saved")


# ════════════════════════════════════════════════════════════
# 3.  Balanced vs Unbalanced paired comparison
# ════════════════════════════════════════════════════════════
# For each model that has both modes, show the delta
models_with_both = df.groupby("model").filter(lambda g: g["mode"].nunique() == 2)["model"].unique()
pairs = []
for m in models_with_both:
    sub = df[df["model"] == m]
    unb = sub[sub["mode"] == "unbalanced"].iloc[0]
    bal = sub[sub["mode"] == "balanced"].iloc[0]
    pairs.append({
        "model": m,
        "pr_auc_unb": unb["pr_auc_mean"],
        "pr_auc_bal": bal["pr_auc_mean"],
        "roc_auc_unb": unb["roc_auc_mean"],
        "roc_auc_bal": bal["roc_auc_mean"],
    })
pdf = pd.DataFrame(pairs).sort_values("pr_auc_unb", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
y_pos = range(len(pdf))

for ax_i, (metric, label) in enumerate([("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]):
    ax = axes[ax_i]
    for i, row in enumerate(pdf.itertuples()):
        u = getattr(row, f"{metric}_unb")
        b = getattr(row, f"{metric}_bal")
        ax.plot([u, b], [i, i], "o-", color="grey", lw=1.5, markersize=6)
        ax.plot(u, i, "o", color="#3274a1", markersize=8, zorder=5)
        ax.plot(b, i, "s", color="#e1812c", markersize=8, zorder=5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(pdf["model"], fontsize=10)
    ax.set_xlabel(label, fontsize=11)
    ax.set_title(f"{label}: Unbalanced vs Balanced", fontsize=12, fontweight="bold")
    ax.legend(handles=[
        plt.Line2D([0],[0], marker="o", color="#3274a1", ls="", markersize=8, label="Unbalanced"),
        plt.Line2D([0],[0], marker="s", color="#e1812c", ls="", markersize=8, label="Balanced"),
    ], loc="lower right", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "step5v2_balanced_vs_unbalanced.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "step5v2_balanced_vs_unbalanced.png"), dpi=150)
plt.close(fig)
print("  [OK] Balanced vs Unbalanced comparison saved")


# ════════════════════════════════════════════════════════════
# 4.  Heatmap: model×mode  vs  metrics
# ════════════════════════════════════════════════════════════
heat_df = df.set_index("label")[["pr_auc_mean", "roc_auc_mean", "f1_mean", "f2_mean"]].copy()
heat_df.columns = ["PR-AUC", "ROC-AUC", "F1@0.5", "F2@0.5"]
heat_df = heat_df.iloc[::-1]  # best on top

fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(heat_df.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, fontsize=10)
ax.set_yticks(range(len(heat_df)))
ax.set_yticklabels(heat_df.index, fontsize=9)
# Annotate cells
for i in range(len(heat_df)):
    for j in range(len(heat_df.columns)):
        val = heat_df.iloc[i, j]
        txt_color = "white" if val > 0.35 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=8, color=txt_color, fontweight="bold")
ax.set_title("Cross-Validation Metrics — 6m Horizon (DEV)", fontsize=12, fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.6, label="Score")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "step5v2_metric_heatmap.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "step5v2_metric_heatmap.png"), dpi=150)
plt.close(fig)
print("  [OK] Metric heatmap saved")


# ════════════════════════════════════════════════════════════
# 5.  Spearman rank correlation between metrics
# ════════════════════════════════════════════════════════════
metrics = ["pr_auc_mean", "roc_auc_mean", "f1_mean", "f2_mean"]
metric_labels = ["PR-AUC", "ROC-AUC", "F1@0.5", "F2@0.5"]
n = len(metrics)
corr_rows = []
for i in range(n):
    for j in range(i + 1, n):
        rho, pval = spearmanr(df[metrics[i]], df[metrics[j]])
        corr_rows.append({
            "metric_a": metric_labels[i],
            "metric_b": metric_labels[j],
            "spearman_rho": round(rho, 4),
            "p_value": round(pval, 6),
        })
corr_df = pd.DataFrame(corr_rows)
corr_path = os.path.join(TAB_DIR, "step5v2_metric_rank_correlation.csv")
corr_df.to_csv(corr_path, index=False)
print(f"  [OK] Spearman rank correlation saved → {corr_path}")
print(corr_df.to_string(index=False))


# ════════════════════════════════════════════════════════════
# 6.  LaTeX table — Tuning comparison
# ════════════════════════════════════════════════════════════
df_sorted = df.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\caption{Optuna-tuned model comparison (6-month horizon, DEV set, GroupKFold $K\!=\!5$, 60 trials). "
             r"Models ranked by PR-AUC. Best value per metric in \textbf{bold}.}")
lines.append(r"\label{tab:step5v2_tuning}")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llcccc}")
lines.append(r"\toprule")
lines.append(r"Model & Mode & PR-AUC & ROC-AUC & F1\textsubscript{@0.5} & F2\textsubscript{@0.5} \\")
lines.append(r"\midrule")

# Find best per column for bold
best_pr  = df_sorted["pr_auc_mean"].max()
best_roc = df_sorted["roc_auc_mean"].max()
best_f1  = df_sorted["f1_mean"].max()
best_f2  = df_sorted["f2_mean"].max()

def fmt_bold(val, best, std=None):
    s = f"{val:.3f}"
    if std is not None:
        s += f" $\\pm$ {std:.3f}"
    if abs(val - best) < 1e-6:
        return r"\textbf{" + s + "}"
    return s

for _, row in df_sorted.iterrows():
    model = row["model"].replace("_", r"\_")
    mode  = row["mode"]
    pr    = fmt_bold(row["pr_auc_mean"], best_pr, row["pr_auc_std"])
    roc   = fmt_bold(row["roc_auc_mean"], best_roc)
    f1    = fmt_bold(row["f1_mean"], best_f1)
    f2    = fmt_bold(row["f2_mean"], best_f2)
    lines.append(f"{model} & {mode} & {pr} & {roc} & {f1} & {f2} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex_path = os.path.join(TAB_DIR, "step5v2_tuning_comparison.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"  [OK] LaTeX tuning table saved → {tex_path}")


# ════════════════════════════════════════════════════════════
# 7.  FVC Missingness — LaTeX table  +  forest plot
# ════════════════════════════════════════════════════════════
miss = pd.read_csv(FVC_MISSINGNESS)
miss6 = miss[miss["horizon"] == "6m"].copy()

# LaTeX table
lines2 = []
lines2.append(r"\begin{table}[ht]")
lines2.append(r"\centering")
lines2.append(r"\caption{FVC missingness analysis (6-month cohort). Mann--Whitney U for continuous variables, "
              r"$\chi^2$ for categorical. Effect sizes: Cohen's $d$ (continuous) and Cramér's $V$ (categorical). "
              r"Significant at $\alpha = 0.05$ marked with $\ast$.}")
lines2.append(r"\label{tab:fvc_missingness}")
lines2.append(r"\small")
lines2.append(r"\begin{tabular}{lcccc}")
lines2.append(r"\toprule")
lines2.append(r"Variable & Test & $p$-value & Effect size & Sig. \\")
lines2.append(r"\midrule")

for _, row in miss6.iterrows():
    var  = row["variable"]
    test = row["test"].replace("Mann-Whitney U", "MW-U").replace("Chi-squared", r"$\chi^2$")
    pval = row["p_value"]
    if pval < 0.001:
        pstr = "$<$0.001"
    else:
        pstr = f"{pval:.3f}"
    eff  = f"{row['effect_size']:.3f}"
    etype = row["effect_type"].replace("Cohen d", "$d$").replace("Cramer V", "$V$")
    sig  = r"$\ast$" if row["significant_005"] else ""
    lines2.append(f"{var} & {test} & {pstr} & {eff} ({etype}) & {sig} \\\\")

lines2.append(r"\bottomrule")
lines2.append(r"\end{tabular}")
lines2.append(r"\end{table}")

tex2_path = os.path.join(TAB_DIR, "step5v2_fvc_missingness.tex")
with open(tex2_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines2))
print(f"  [OK] FVC missingness LaTeX table saved → {tex2_path}")

# Forest-style plot for effect sizes
fig, ax = plt.subplots(figsize=(8, 5))
miss6_sorted = miss6.sort_values("effect_size", ascending=True).reset_index(drop=True)
colors_miss = ["#d62728" if s else "#2ca02c" for s in miss6_sorted["significant_005"]]
y = range(len(miss6_sorted))
ax.barh(list(y), miss6_sorted["effect_size"], color=colors_miss, edgecolor="white", height=0.6)
ax.set_yticks(list(y))
ax.set_yticklabels(miss6_sorted["variable"], fontsize=9)
ax.set_xlabel("Effect Size (Cohen's $d$ / Cramér's $V$)", fontsize=11)
ax.set_title("FVC Missingness: Effect Sizes (6m cohort)", fontsize=12, fontweight="bold")
ax.axvline(x=0.2, ls="--", color="grey", alpha=0.5, lw=0.8, label="Small effect (0.2)")
ax.legend(handles=[
    Patch(color="#d62728", label="Significant ($p < 0.05$)"),
    Patch(color="#2ca02c", label="Not significant"),
    plt.Line2D([0],[0], ls="--", color="grey", label="Small effect boundary"),
], loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "step5v2_fvc_missingness_forest.pdf"), dpi=150)
fig.savefig(os.path.join(FIG_DIR, "step5v2_fvc_missingness_forest.png"), dpi=150)
plt.close(fig)
print("  [OK] FVC missingness forest plot saved")


# ════════════════════════════════════════════════════════════
# Copy to overleaf
# ════════════════════════════════════════════════════════════
import shutil
for f in os.listdir(FIG_DIR):
    if f.endswith(".png"):
        shutil.copy2(os.path.join(FIG_DIR, f), os.path.join(OVERLEAF_FIG, f))
for f in os.listdir(TAB_DIR):
    if f.endswith(".tex"):
        shutil.copy2(os.path.join(TAB_DIR, f), os.path.join(OVERLEAF_TAB, f))
print("  → Copied figures/tables to overleaf/images/")


# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  All artefacts generated in {elapsed:.1f}s")
print(f"{'='*60}")
print(f"""
  Figures:
    {FIG_DIR}/step5v2_prauc_barchart.pdf
    {FIG_DIR}/step5v2_balanced_vs_unbalanced.pdf
    {FIG_DIR}/step5v2_metric_heatmap.pdf
    {FIG_DIR}/step5v2_fvc_missingness_forest.pdf

  Tables (LaTeX):
    {tex_path}
    {tex2_path}

  Analysis:
    {corr_path}
""")
