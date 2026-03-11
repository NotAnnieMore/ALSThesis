"""
Step 5 — Ablation v2: evaluate the best model configs with incremental feature blocks.

Uses the best hyperparameters from the Optuna tuning (step5_tuning/*.json)
and evaluates them with progressively larger feature sets via GroupKFold CV
(no re-tuning — just feature ablation on pre-tuned params).

Feature blocks (cumulative):
  A : Baseline only     (demographics + ALSFRS-R items)
  B : A + Vitals
  C : A + FVC
  D : A + Vitals + FVC
  E : D + Treatment     (riluzole_pre_t0, study_arm, n_conmeds_pre_t0)
  F : D + Treatment + Labs  (alt_t0, creatinine_t0)      ← full v2

Models evaluated: top performers from tuning — XGB(unb), RF(unb), LGBM(bal), LR(bal)

Outputs
-------
  04_outputs/tables/step5v2_ablation.csv
  04_outputs/figures/step5v2_ablation_prauc.pdf
"""

from __future__ import annotations
import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, fbeta_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── paths ──
ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED  = os.path.join(ROOT, "01_data", "processed")
TAB_DIR    = os.path.join(ROOT, "04_outputs", "tables", "step5_ablation")
FIG_DIR    = os.path.join(ROOT, "04_outputs", "figures", "step5_ablation")
TUNING_DIR = os.path.join(ROOT, "04_outputs", "tables", "step5_tuning")
OVERLEAF_FIG = os.path.join(ROOT, "overleaf", "images", "figures")
OVERLEAF_TAB = os.path.join(ROOT, "overleaf", "images", "tables")

HORIZON    = "6m"
N_SPLITS   = 5
RAPID_FRAC = 0.30

# ── Feature block definitions ──
# Columns that are NEVER features
META_COLS = {
    "subject_id", "t0_delta_days", "vitals_delta_days", "fvc_delta_days",
    "slope_90d_per_30d", "slope_180d_per_30d", "ALSFRS_Responded_By",
}

# Block membership for non-meta columns
VITALS_COLS = {
    "Weight_kg_t0", "Height_cm_t0", "BMI_t0", "Pulse", "Respiratory_Rate",
    "Temperature", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
    "Baseline_Standing_BP_Systolic", "Baseline_Standing_BP_Diastolic",
    "Baseline_Supine_BP_Systolic", "Baseline_Supine_BP_Diastolic",
    "vitals_delta_days",
}

FVC_COLS = {
    "FVC_Liters_best_t0", "FVC_pctNormal_best_t0", "subject_normal",
    "fvc_delta_days",
}

TREATMENT_COLS = {
    "riluzole_pre_t0", "study_arm", "n_conmeds_pre_t0",
}

LABS_COLS = {
    "alt_t0", "creatinine_t0",
}

# Block configs (cumulative)
BLOCKS = {
    "A: Baseline":               lambda all_c: [c for c in all_c if c not in META_COLS | VITALS_COLS | FVC_COLS | TREATMENT_COLS | LABS_COLS],
    "B: +Vitals":                lambda all_c: [c for c in all_c if c not in META_COLS | FVC_COLS | TREATMENT_COLS | LABS_COLS],
    "C: +FVC":                   lambda all_c: [c for c in all_c if c not in META_COLS | VITALS_COLS | TREATMENT_COLS | LABS_COLS],
    "D: +Vitals+FVC":            lambda all_c: [c for c in all_c if c not in META_COLS | TREATMENT_COLS | LABS_COLS],
    "E: +Treatment":             lambda all_c: [c for c in all_c if c not in META_COLS | LABS_COLS],
    "F: Full v2":                lambda all_c: [c for c in all_c if c not in META_COLS],
}

# Models to evaluate (key, balanced, needs_scaling)
MODELS = [
    ("XGB_unbalanced",  "XGB",  False, False),
    ("RF_unbalanced",   "RF",   False, False),
    ("LGBM_balanced",   "LGBM", True,  False),
    ("LR_balanced",     "LR",   True,  True),
]


def load_best_params(model_key: str, mode: str) -> dict:
    tag = f"{model_key}_{mode}_{HORIZON}_best.json"
    path = os.path.join(TUNING_DIR, tag)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["full_params"]


def build_preprocessor(X, scale):
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if scale:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")


CLF_CLASS = {
    "XGB": XGBClassifier,
    "RF":  RandomForestClassifier,
    "LGBM": LGBMClassifier,
    "LR":  LogisticRegression,
}


def build_clf(model_key, params, balanced, n_pos=0, n_neg=0):
    p = dict(params)
    if balanced:
        if model_key in ("RF", "LR"):
            p["class_weight"] = "balanced"
        elif model_key == "LGBM":
            p["is_unbalance"] = True
        elif model_key == "XGB" and n_pos > 0:
            p["scale_pos_weight"] = n_neg / n_pos
    return CLF_CLASS[model_key](**p)


def cv_evaluate(model_key, params, balanced, X, slope, groups, scale):
    preproc_template = build_preprocessor(X, scale)
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = {"pr_auc": [], "roc_auc": [], "f1": [], "f2": []}

    for tr_idx, va_idx in gkf.split(X, groups=groups):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        slope_tr, slope_va = slope.iloc[tr_idx], slope.iloc[va_idx]
        thr = float(np.nanquantile(slope_tr.values, RAPID_FRAC))
        ytr = (slope_tr <= thr).astype(int)
        yva = (slope_va <= thr).astype(int)

        n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
        clf = build_clf(model_key, params, balanced, n_pos, n_neg)
        preproc = clone(preproc_template)
        pipe = Pipeline([("prep", preproc), ("clf", clf)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr, ytr)

        proba = pipe.predict_proba(Xva)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        fold_results["pr_auc"].append(average_precision_score(yva, proba))
        fold_results["roc_auc"].append(roc_auc_score(yva, proba))
        fold_results["f1"].append(f1_score(yva, y_pred, zero_division=0))
        fold_results["f2"].append(fbeta_score(yva, y_pred, beta=2, zero_division=0))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in fold_results.items()}


def main():
    t_global = time.time()

    # Load data
    ds_path = os.path.join(PROCESSED, f"dataset_{HORIZON}_v2.csv")
    hs_path = os.path.join(PROCESSED, f"holdout_split_{HORIZON}.csv")
    df = pd.read_csv(ds_path)
    hs = pd.read_csv(hs_path)
    dev_ids = set(hs.loc[hs["partition"] == "dev", "subject_id"])
    df = df[df["subject_id"].isin(dev_ids)].copy()

    slope_col = "slope_180d_per_30d"
    df[slope_col] = pd.to_numeric(df[slope_col], errors="coerce")
    df = df.dropna(subset=[slope_col])

    slope  = df[slope_col].reset_index(drop=True)
    groups = df["subject_id"].reset_index(drop=True)
    all_cols = [c for c in df.columns if c not in {"subject_id", slope_col, "slope_90d_per_30d"}]

    print(f"DEV set: {len(df)} subjects, {len(all_cols)} total columns")

    results = []

    for model_label, model_key, balanced, scale in MODELS:
        mode = "balanced" if balanced else "unbalanced"
        params = load_best_params(model_key, mode)
        print(f"\n{'='*60}")
        print(f"  {model_label}")
        print(f"{'='*60}")

        for block_name, col_fn in BLOCKS.items():
            t0 = time.time()
            feat_cols = col_fn(all_cols)
            X = df[feat_cols].reset_index(drop=True)
            n_feats = X.shape[1]

            metrics = cv_evaluate(model_key, params, balanced, X, slope, groups, scale)
            elapsed = time.time() - t0

            row = {
                "model": model_label,
                "block": block_name,
                "n_features": n_feats,
                "pr_auc_mean": round(metrics["pr_auc"][0], 4),
                "pr_auc_std": round(metrics["pr_auc"][1], 4),
                "roc_auc_mean": round(metrics["roc_auc"][0], 4),
                "f1_mean": round(metrics["f1"][0], 4),
                "f2_mean": round(metrics["f2"][0], 4),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(row)
            print(f"  {block_name:20s}  n={n_feats:3d}  PR-AUC={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f}  ({elapsed:.1f}s)")

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(TAB_DIR, "step5v2_ablation.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path}")

    # ── Figure: grouped bar chart ──
    blocks_order = list(BLOCKS.keys())
    model_labels = [m[0] for m in MODELS]
    n_models = len(model_labels)
    n_blocks = len(blocks_order)
    x = np.arange(n_blocks)
    width = 0.8 / n_models

    colors = ["#3274a1", "#e1812c", "#3a923a", "#c03d3e"]
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, ml in enumerate(model_labels):
        sub = res_df[res_df["model"] == ml].set_index("block").reindex(blocks_order)
        ax.bar(x + i * width, sub["pr_auc_mean"], width, yerr=sub["pr_auc_std"],
               label=ml, color=colors[i], capsize=2, edgecolor="white")

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([b.split(":")[0] for b in blocks_order], fontsize=10)
    ax.set_ylabel("PR-AUC (mean ± std)", fontsize=11)
    ax.set_title("Feature Block Ablation — 6m Horizon (DEV, Optuna-tuned params)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.30, 0.50)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "step5v2_ablation_prauc.pdf"), dpi=150)
    fig.savefig(os.path.join(FIG_DIR, "step5v2_ablation_prauc.png"), dpi=150)
    plt.close(fig)

    # ── LaTeX table ──
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Feature block ablation (6m, DEV). PR-AUC with Optuna-tuned hyperparameters. "
                 r"Blocks are cumulative: each row adds new features to the previous.}")
    lines.append(r"\label{tab:ablation_v2}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Block & $n$ & XGB\textsubscript{unb} & RF\textsubscript{unb} "
                 r"& LGBM\textsubscript{bal} & LR\textsubscript{bal} \\")
    lines.append(r"\midrule")

    for block in blocks_order:
        sub = res_df[res_df["block"] == block]
        n_feats = sub.iloc[0]["n_features"]
        vals = []
        for ml in model_labels:
            row = sub[sub["model"] == ml].iloc[0]
            vals.append(f"{row['pr_auc_mean']:.3f}")
        lines.append(f"{block} & {n_feats} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(TAB_DIR, "step5v2_ablation.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX → {tex_path}")

    # Copy to overleaf
    import shutil
    for fl in os.listdir(FIG_DIR):
        if fl.endswith(".png"):
            shutil.copy2(os.path.join(FIG_DIR, fl), os.path.join(OVERLEAF_FIG, fl))
    for fl in os.listdir(TAB_DIR):
        if fl.endswith(".tex"):
            shutil.copy2(os.path.join(TAB_DIR, fl), os.path.join(OVERLEAF_TAB, fl))
    print("  → Copied figures/tables to overleaf/images/")

    elapsed_total = time.time() - t_global
    print(f"\n  Total ablation time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
